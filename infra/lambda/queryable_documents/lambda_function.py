import json
import os
import boto3

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    # Handle CORS OPTIONS request
    http_method = event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod")
    if http_method == "OPTIONS":
        return {
            "statusCode": 204,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": ""
        }

    bucket_name = os.environ.get("BUCKET_NAME", "ayan-deaws-lab-600743178533")
    prefix = os.environ.get("FAISS_METADATA_PREFIX", "embeddings/faiss/")

    try:
        # Paginate S3 list_objects_v2
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        metadata_files = []
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj["Key"].endswith(".metadata.json"):
                        metadata_files.append(obj)

        # Sort by LastModified descending
        metadata_files.sort(key=lambda x: x["LastModified"], reverse=True)

        results = []
        for obj in metadata_files:
            try:
                res = s3_client.get_object(Bucket=bucket_name, Key=obj["Key"])
                body_content = res["Body"].read().decode("utf-8")
                data = json.loads(body_content)
                
                doc_name = obj["Key"].split("/")[-1].replace(".metadata.json", "")
                
                results.append(
                    {
                        "doc_name": doc_name,
                        "source_key": data.get("source_key"),
                        "source_version_id": data.get("source_version_id"),
                        "chunk_count": data.get("chunk_count"),
                        "embedding_model": data.get("embedding_model"),
                        "chunks_s3_key": data.get("chunks_s3_key"),
                        "faiss_index_s3_key": data.get("faiss_index_s3_key"),
                        "created_at": data.get("created_at"),
                        "status": data.get("status")
                    }
                )
            except Exception as e:
                print(f"Failed to read metadata {obj['Key']}: {e}")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps(results)
        }

    except Exception as e:
        print(f"Error fetching queryable documents: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": "Internal Server Error"})
        }
