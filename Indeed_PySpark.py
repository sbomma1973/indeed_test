import sys
from collections.abc import Iterator
from typing import Any, Literal, cast, Iterable

import boto3
import json
import asyncio
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, Row
from pyspark.sql.functions import col, struct, collect_list, explode
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError
from itertools import islice
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IndexMode = Literal["DEFAULT", "WITH_EMBEDDING"]

def get_glue_args(mandatory_fields: list[str], default_optional_args: dict = {}) -> dict[str, str]:
    given_optional_fields_key = list(set([i.removeprefix('--') for i in sys.argv]).intersection([i for i in default_optional_args]))
    resolved_args = getResolvedOptions(sys.argv, mandatory_fields + given_optional_fields_key)
    return { **default_optional_args, **resolved_args }

# Initialize Spark with Elasticsearch configuration
def create_spark_session(sink_creds):
    spark = (SparkSession.builder
        .appName("Indeed_ES_Indexer")
        .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.3")
        .config("es.nodes", sink_creds['endpoint'].split('://')[1])
        .config("es.port", "443")
        .config("es.nodes.wan.only", "true")
        .config("es.net.http.auth.user", sink_creds['username'])
        .config("es.net.http.auth.pass", sink_creds['password'])
        .config("es.batch.size.entries", "1000")
        .config("es.batch.size.bytes", "5mb")
        .config("es.batch.write.retry.count", "5")
        .config("es.batch.write.retry.wait", "10s")
        .config("es.http.timeout", "5m")
        .config("es.http.retries", "5")
        .config("es.nodes.wan.only", "true")
        .config("es.write.operation", "upsert")
        .config("es.mapping.id", "accountId")
        .config("es.write.operation.type", "external")
        .getOrCreate())
    return spark

# Get arguments and initialize
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

# Initialize AWS services
aws_session = boto3.session.Session()
secrets_client = aws_session.client(service_name='secretsmanager', region_name='us-east-2')
s3_client = aws_session.client(service_name='s3')
sts_client = aws_session.client('sts')
logger.info('AWS session established')

# Get credentials
creds_response = secrets_client.get_secret_value(SecretId=args['credentials'])
creds = json.loads(creds_response['SecretString'])
source_creds = creds[args['source']]
sink_creds = creds[args['sink']]
logger.info('Credentials retrieved from Secrets Manager')

# Initialize Spark context
spark = create_spark_session(sink_creds)
sc = spark.sparkContext
glueContext = GlueContext(sc)
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
logger.info('Spark session initialized with Elasticsearch configuration')

# Load schema
schema_object = s3_client.get_object(Bucket=args['bucket'], Key=args['source_schema'])
schema_string = schema_object['Body'].read().decode('utf-8')
schema = StructType.fromJson(json.loads(schema_string))
logger.info('Document schema loaded')

def refreshable_assumed_role_session():
    client = boto3.client('sts')
    
    def refresh():
        logger.info('Refreshing tokens for assume role.')
        assumed_role = client.assume_role(
            RoleArn=args['sagemaker_role_arn'],
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

    session_credentials = RefreshableCredentials.create_from_metadata(
        metadata=refresh(),
        refresh_using=refresh,
        method='sts-assume-role'
    )

    session = get_session()
    session._credentials = session_credentials
    session.set_config_variable("region", 'us-east-2')
    return boto3.Session(botocore_session=session)

def prepare_document(row):
    """Prepare document for Elasticsearch indexing"""
    doc = row.esDocument.asDict(True)
    # Remove fields we don't want to index
    doc.pop('allJobDescs', "")
    doc.pop('allNormalizedJobTitles', "")
    return doc

def process_with_embeddings(df, sagemaker_endpoint):
    """Process documents with embeddings from SageMaker"""
    boto3_session = refreshable_assumed_role_session()
    sagemaker_client = boto3_session.client('sagemaker-runtime')
    
    def get_embeddings(batch_iter):
        batch = list(batch_iter)
        resumes = [map_resume(row) for row in batch]
        try:
            response = sagemaker_client.invoke_endpoint(
                EndpointName=sagemaker_endpoint,
                ContentType='application/json',
                Body=json.dumps({'resumes': resumes})
            )
            vectors = json.loads(response['Body'].read().decode())
            return vectors
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return [None] * len(batch)

    # Process in batches
    batch_size = int(args['sagemaker_batch_size'])
    df_with_embeddings = df.mapInPandas(
        lambda iter: get_embeddings(iter),
        schema=["resumeVector"]
    )
    
    return df_with_embeddings

def map_resume(row):
    """Map document to resume format for embedding"""
    esDoc = row.esDocument.asDict(True)
    allAttrs = esDoc.get('allAttrs') or {}
    
    return {
        'accountId': esDoc['accountId'],
        'experiences': map_experience(esDoc),
        'summary': esDoc.get('summary') or "",
        'additionalInfo': esDoc.get('additionalInfo') or "",
        'skills': [map_skill(x) for x in (allAttrs.get('skills') or [])],
        'educations': [map_education(x) for x in (allAttrs.get('educations') or [])],
        'occupationNames': [(x['label'] or "") for x in (allAttrs.get('occupations') or [])],
        'certificationsTitles': [(x['label'] or "") for x in (allAttrs.get('certifications') or [])]
    }

def map_skill(skill):
    return {
        'label': skill.get('label') or "",
        'monthsOfExperience': skill.get('monthsOfExperience') or 0
    }

def map_experience(esDoc):
    allAttrs = esDoc.get('allAttrs') or {}
    experiences = allAttrs.get('experiences') or []
    
    return [{
        'title': exp.get('title') or "",
        'company': exp.get('company') or "",
        'description': exp.get('description') or "",
        'startDate': exp.get('startDate'),
        'endDate': exp.get('endDate'),
        'isCurrent': exp.get('isCurrent') or False
    } for exp in experiences]

def map_education(education):
    return {
        'degree': education.get('degree') or "",
        'field': education.get('field') or "",
        'school': education.get('school') or ""
    }

def reindex_documents(state_list: list[str], country_list: list[str] | None, from_date: str | None, to_date: str | None, index_mode: IndexMode) -> None:
    """Main function to reindex documents"""
    try:
        # Read source data into DataFrame
        source_df = spark.read.format("mongo") \
            .option("uri", source_creds['uri']) \
            .option("database", args['source_db_name']) \
            .option("collection", args['source_db_collection']) \
            .load()

        # Apply filters
        filtered_df = source_df.filter(
            (col("state").isin(state_list)) &
            (col("country").isin(country_list if country_list else state_list))
        )
        
        if from_date:
            filtered_df = filtered_df.filter(col("lastModifiedDate") >= from_date)
        if to_date:
            filtered_df = filtered_df.filter(col("lastModifiedDate") <= to_date)

        # Process with embeddings if required
        if index_mode == "WITH_EMBEDDING":
            filtered_df = process_with_embeddings(filtered_df, args['sagemaker_endpoint'])

        # Prepare documents for Elasticsearch
        es_ready_df = filtered_df.select(
            col("accountId"),
            col("version"),
            struct([prepare_document(col("*"))]).alias("doc")
        )

        # Write to Elasticsearch with optimized settings
        (es_ready_df.write
            .format("org.elasticsearch.spark.sql")
            .option("es.resource", f"{args['sink_index']}")
            .option("es.mapping.id", "accountId")
            .option("es.write.operation", "upsert")
            .mode("append")
            .save())

        logger.info("Indexing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Parse states and countries
        states = args['states'].split(',')
        countries = args['countries'].split(',') if args['countries'] else None
        
        # Start indexing
        reindex_documents(
            state_list=states,
            country_list=countries,
            from_date=args['from_date'],
            to_date=args['to_date'],
            index_mode=cast(IndexMode, args['index_mode'])
        )
        
        job.commit()
    except Exception as e:
        logger.error(f"Job failed: {str(e)}", exc_info=True)
        sys.exit(1)