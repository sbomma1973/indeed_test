import sys
from collections.abc import Iterator
from typing import Any

import boto3
import json
import asyncio
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session
from elasticsearch import Elasticsearch
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from elasticsearch.helpers import bulk
from pyspark.context import SparkContext
from pyspark.sql.types import StructType, Row
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError
from typing import Literal, cast
from typing import Iterable
from itertools import islice

IndexMode = Literal["DEFAULT", "WITH_EMBEDDING"]

def get_glue_args(mandatory_fields: list[str], default_optional_args: dict = {}) -> dict[str, str]:
    given_optional_fields_key = list(set([i.removeprefix('--') for i in sys.argv]).intersection([i for i in default_optional_args]))
    resolved_args = getResolvedOptions(sys.argv, mandatory_fields + given_optional_fields_key)
    return { **default_optional_args, **resolved_args }

# use get_glue_args instead of calling getResolvedOptions directly
args = get_glue_args([
    'JOB_NAME',
    'credentials',
    'bucket',
    'batch_size',
    'states',
    'countries',
    'source',
    'source_schema',
    'source_db_name',
    'source_db_collection',
    'sink',
    'sink_index',
    'mode',
    'sagemaker_endpoint',
    'sagemaker_role_arn',
],{
    'from_date': None,
    'to_date': None,
    'index_mode': 'DEFAULT',
    'sagemaker_batch_size': '100'
})

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
print('Job initialized')

aws_session = boto3.session.Session()
secrets_client = aws_session.client(service_name='secretsmanager', region_name='us-east-2')
s3_client = aws_session.client(service_name='s3')
sts_client = aws_session.client('sts')
print('AWS session established')

sagemaker_role_arn = args['sagemaker_role_arn']
sagemaker_endpoint = args['sagemaker_endpoint']

creds_response = secrets_client.get_secret_value(SecretId=args['credentials'])
creds = json.loads(creds_response['SecretString'])
source_creds = creds[args['source']]
sink_creds = creds[args['sink']]
print('Credentials retrieved from Secrets Manager')

schema_object = s3_client.get_object(Bucket=args['bucket'], Key=args['source_schema'])
schema_string = schema_object['Body'].read().decode('utf-8')
schema = StructType.fromJson(json.loads(schema_string))
print('Document schema loaded')

worker_vars = sc.broadcast({
    'endpoint': sink_creds['endpoint'],
    'username': sink_creds['username'],
    'password': sink_creds['password'],
    'index': args['sink_index'],
    'batch_size': args['batch_size']
})
docs_indexed = sc.accumulator(0)
version_conflicts = sc.accumulator(0)
docs_failed = sc.accumulator(0)
print('Spark worker variables initialized')

def refreshable_assumed_role_session():
    client = boto3.client('sts')

    def refresh():
        print('Refreshing tokens for assume role.')
        assumed_role = client.assume_role(
            RoleArn=sagemaker_role_arn,
            RoleSessionName=args['JOB_NAME'],
            DurationSeconds=3600
        )
        assumed_credentials = assumed_role['Credentials']
        return {
            'access_key': assumed_credentials['AccessKeyId'],
            'secret_key': assumed_credentials['SecretAccessKey'],
            'token': assumed_credentials['SessionToken'],
            'expiry_time': assumed_credentials['Expiration'].isoformat()
        }

    # Create a new botocore session with refreshable credentials
    session_credentials = RefreshableCredentials.create_from_metadata(
        metadata=refresh(),
        refresh_using=refresh,
        method='sts-assume-role'
    )

    session = get_session()
    session._credentials = session_credentials
    session.set_config_variable("region", 'us-east-2')
    boto3_session = boto3.Session(botocore_session=session)

    return boto3_session

# index original esDocument
def index_partition(partition: Iterator[Row]) -> None:
    es = Elasticsearch(hosts=worker_vars.value['endpoint'],
                       http_auth=(worker_vars.value['username'], worker_vars.value['password']),
                       max_retries=3,
                       retry_on_timeout=True)
    batch_size = int(worker_vars.value['batch_size'])

    for batch in batched(partition, batch_size):
        actions = []
        for doc in batch:
            # allJobDescs and allNormalizedJobTitles are fields that we don't want to add to the ES index,
            # however the bulk action in the ES library will ignore the dynamic setting in the ES mapping and index them as long as they are present,
            # so we need to remove them manually
            source = doc.esDocument.asDict(True)
            source.pop('allJobDescs', "")
            source.pop('allNormalizedJobTitles', "")
            actions.append({
                '_index': worker_vars.value['index'],
                '_id': doc.accountId,
                'version_type': 'external',
                'version': doc.version,
                '_source': source
            })
        index_batch(es, actions)

# send esDocs to sagemaker endpoint to get embeddings first and then index embeddings with original esDocument together
def index_partition_with_embedding(partition: Iterator[Row]) -> None:
    es = Elasticsearch(hosts=worker_vars.value['endpoint'],
                       http_auth=(worker_vars.value['username'], worker_vars.value['password']),
                       max_retries=3,
                       retry_on_timeout=True)
    boto3_session = refreshable_assumed_role_session()
    sagemaker_client = boto3_session.client('sagemaker-runtime')

    batch_size = int(worker_vars.value['batch_size'])

    with ThreadPoolExecutor(max_workers=4) as tp:
        for batch in batched(partition, batch_size):
            actions = asyncio.run(map_actions(tp, sagemaker_client, batch))
            index_batch(es, actions)

def batched(it, n: int):
    if n < 1:
        raise ValueError('n must be at least one')
    while batch := tuple(islice(it, n)):
        yield batch

async def map_actions(tp, sagemaker_client, docs):
    tasks = []
    sagemaker_endpoint_batch_size = int(args['sagemaker_batch_size'])
    for batch in batched(iter(docs), sagemaker_endpoint_batch_size):
        tasks.append(map_actions_batch(tp, sagemaker_client, batch))

    # Gather all asynchronous tasks
    results = await asyncio.gather(*tasks)
    actions = [item for row in results for item in row]

    return actions

async def map_actions_batch(tp, sagemaker_client, docs):
    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(tp, invoke_sagemaker_endpoint, sagemaker_client, docs)

    actions = []
    for i in range(len(docs)):
        source = docs[i].esDocument.asDict(True)
        source['resumeVector'] = vectors[i]
        source.pop('allJobDescs', "")
        source.pop('allNormalizedJobTitles', "")
        actions.append({
            '_index': worker_vars.value['index'],
            '_id': docs[i].accountId,
            'version_type': 'external',
            'version': docs[i].version,
            '_source': source
        })
    return actions

def invoke_sagemaker_endpoint(sagemaker_client, docs):
    resumes = map_resumes(docs)
    payload = {
        'resumes': resumes
    }
    try:
        response = sagemaker_client.invoke_endpoint(
            EndpointName=sagemaker_endpoint,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        result = response['Body'].read().decode()
        return json.loads(result)
    except ClientError as e:
        print(f"Received error from SageMaker endpoint: {e.response['Error']['Message']}")
        print(f"Failed resumes: {[x['accountId'] for x in resumes]}")
        raise

def map_resumes(docs):
    resumes = []
    for doc in docs:
        esDoc = doc.esDocument.asDict(True)
        allAttrs = esDoc.get('allAttrs') or {}

        resume = {
            'accountId': esDoc['accountId'],
            'experiences': map_experience(esDoc),
            'summary': esDoc.get('summary') or "",
            'additionalInfo': esDoc.get('additionalInfo') or "",
            'skills': [map_skill(x) for x in (allAttrs.get('skills') or [])],
            'educations': [map_education(x) for x in (allAttrs.get('educations') or [])],
            'occupationNames': [(x['label'] or "") for x in (allAttrs.get('occupations') or [])],
            'certificationsTitles': [(x['label'] or "") for x in (allAttrs.get('certifications') or [])]
        }

        resumes.append(resume)

    return resumes

def map_skill(skill):
    return {
        'label': skill.get('label') or "",
        'monthsOfExperience': skill.get('monthsOfExperience') or 0
    }

def map_experience(esDoc):
    experiences = []
    account_id = esDoc['accountId']

    allJobDescs = esDoc.get('allJobDescs') or []
    allNormalizedJobTitles = esDoc.get('allNormalizedJobTitles') or []
    anycompany = esDoc.get('anycompany') or []

    if len(allJobDescs) != len(allNormalizedJobTitles):
        print(f'allJobDescs size not match allNormalizedJobTitles for accountId {account_id}')
    else:
        for i in range(len(allJobDescs)):
            experiences.append({ 'jobDescription': allJobDescs[i], 'jobTitle': allNormalizedJobTitles[i], 'company': "" })

    if not experiences:
        for company in anycompany:
            experiences.append({ 'company': company, 'jobDescription': "", "jobTitle": "" })
    elif len(anycompany) == len(experiences):
        for i in range(len(anycompany)):
            experiences[i]['company'] = esDoc['anycompany'][i]

    return experiences

def map_education(education):
    return {
        'degreeName': education.get('label') or "",
        'schoolName': education.get('schoolName') or "",
        'fieldOfStudy': education.get('fieldOfStudy') or ""
    }

def index_batch(es: Elasticsearch, actions: list[Any]) -> None:
    global docs_indexed
    global version_conflicts
    global docs_failed

    response = bulk(client=es, actions=actions, raise_on_error=False, max_retries=3)
    for error in response[1]:
        account_id = error['index']['_id']
        if error['index']['status'] == 409:
            print(f'Version conflict: {account_id}')
            version_conflicts += 1
        else:
            print(f'Indexing failure: {account_id}')
            print(error)
            docs_failed += 1
    docs_indexed += response[0]


def reindex_documents(state_list: list[str], country_list: list[str] | None, from_date: str | None, to_date: str | None, index_mode: IndexMode) -> None:
    print(f'Reindexing started for state(s) {state_list}, country(ies) {country_list}, index_mode {index_mode} and between lastModified {from_date} and {to_date}')
    dataframe = (spark.read
                 .format('mongodb')
                 .schema(schema)
                 .option('spark.mongodb.read.database', args['source_db_name'])
                 .option('spark.mongodb.read.collection', args['source_db_collection'])
                 .option('spark.mongodb.read.connection.uri', source_creds['connectionString'])
                 .option('spark.mongodb.read.comment', 'glue_reindexing_job_run')
                 .option("spark.mongodb.input.connectionTimeoutMS", "60000")
                 .load())
    dataframe = dataframe.filter(dataframe['esDocument.state'].isin(state_list))
    if country_list is not None:
        if 'ROW' in country_list:
            dataframe = dataframe.filter(dataframe['esDocument.country'] != 'US')
        else:
            dataframe = dataframe.filter(dataframe['esDocument.country'].isin(country_list))
    if from_date is not None:
        dataframe = dataframe.filter((dataframe['esDocument.lastModified'] >= int(from_date)) | (dataframe['esDocument.jsLastAppliedDate'] >= int(from_date)))
    if to_date is not None:
            dataframe = dataframe.filter((dataframe['esDocument.lastModified'] < int(to_date)) | (dataframe['esDocument.jsLastAppliedDate'] < int(to_date)))

    if args['mode'] == 'dry':
        dataframe.write.mode('overwrite').format('noop').save()
    elif args['mode'] == 'fragile':
        # The following cannot tolerate any version conflicts or indexing failures without failing the entire job run
        dataframe = dataframe.select('version', 'esDocument.*')
        (dataframe.write
         .mode('append')
         .format('org.elasticsearch.spark.sql')
         .option('es.nodes', sink_creds['endpoint'])
         .option('es.net.http.auth.user', sink_creds['username'])
         .option('es.net.http.auth.pass', sink_creds['password'])
         .option('es.port', 443)
         .option('es.nodes.wan.only', True)
         .option('es.index.auto.create', False)
         .option('es.resource', args['sink_index'])
         .option('es.mapping.id', 'accountId')
         .option('es.mapping.version', 'version')
         .option('es.mapping.exclude', 'version')
         .option('es.write.operation', 'index')
         .save())
    elif index_mode == 'WITH_EMBEDDING':
        dataframe.foreachPartition(index_partition_with_embedding)
    else:
        dataframe.foreachPartition(index_partition)
    print(f'Reindexing finished for state(s) {state_list}, country(ies) {country_list} and between lastModified {from_date} and {to_date}')


states = str(args['states']).split(',')
countries = None
from_date = args['from_date']
to_date = args['to_date']
index_mode = cast(IndexMode, args['index_mode'])

if args['countries'] != 'all':
    countries = str(args['countries']).split(',')

reindex_documents(states, countries, from_date, to_date, index_mode)

print('Job complete')
if args['mode'] == 'default':
    print(f'Docs indexed: {docs_indexed.value}')
    print(f'Version conflicts: {version_conflicts.value}')
    print(f'Doc indexing failures: {docs_failed.value}')

worker_vars.destroy()
job.commit()
