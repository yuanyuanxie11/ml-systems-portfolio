# ECS Fargate — churn pipeline task

Use an **ECR** image built from this repository’s [`Dockerfile`](../Dockerfile). Attach an **ECS task IAM role** with `s3:PutObject` (and optionally `kms:Decrypt` if the bucket uses CMK) on your artifact prefix — do not bake AWS keys into the image.

## Build and push

```bash
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO
docker build -t churn-pipeline:latest .
docker tag churn-pipeline:latest $ECR_REPO/churn-pipeline:latest
docker push $ECR_REPO/churn-pipeline:latest
```

## Configure run

Set in task definition environment (or Secrets Manager for non-secret overrides only if needed):

| Variable       | Purpose                                      |
|----------------|----------------------------------------------|
| `AWS_DEFAULT_REGION` | Region for boto3 S3 client            |
| `S3_BUCKET`    | Overrides `config/churn_pipeline.yaml` `s3.bucket` |

In [`config/churn_pipeline.yaml`](../config/churn_pipeline.yaml), set `s3.upload_enabled: true` for production runs inside AWS.

Evidence of upload: CloudWatch Logs from the container lists `Uploaded: s3://...` lines logged by [`src/churn_pipeline/s3_upload.py`](../src/churn_pipeline/s3_upload.py).

## Example task definition

See [`infra/ecs-task-definition.example.json`](../infra/ecs-task-definition.example.json) — replace `ACCOUNT`, `REGION`, and image URI before registering.
