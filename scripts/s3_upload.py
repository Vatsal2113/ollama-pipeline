#!/usr/bin/env python3
"""
Enhanced S3 Upload Utility for Ollama Pipeline

This script provides robust S3 upload functionality with retry logic,
progress tracking, and better error handling.
"""

import os
import sys
import argparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
import time
from typing import Optional


class S3Uploader:
    """Enhanced S3 uploader with retry logic and progress tracking."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize S3 client."""
        try:
            self.s3_client = boto3.client('s3', region_name=region_name)
            self.region = region_name
        except NoCredentialsError:
            print("[ERROR] AWS credentials not found. Please configure AWS credentials.")
            sys.exit(1)
    
    def upload_file(
        self, 
        local_path: str, 
        bucket: str, 
        s3_key: str, 
        max_retries: int = 3,
        content_type: Optional[str] = None
    ) -> bool:
        """
        Upload a file to S3 with retry logic.
        
        Args:
            local_path: Local file path
            bucket: S3 bucket name
            s3_key: S3 object key
            max_retries: Maximum retry attempts
            content_type: Content type for the file
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(local_path):
            print(f"[ERROR] Local file not found: {local_path}")
            return False
        
        file_size = os.path.getsize(local_path)
        print(f"[*] Uploading {local_path} ({file_size} bytes) to s3://{bucket}/{s3_key}")
        
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        for attempt in range(max_retries):
            try:
                self.s3_client.upload_file(
                    local_path, 
                    bucket, 
                    s3_key, 
                    ExtraArgs=extra_args
                )
                print(f"[✓] Successfully uploaded to s3://{bucket}/{s3_key}")
                return True
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                print(f"[ERROR] Upload attempt {attempt + 1} failed: {error_code} - {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"[*] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] All {max_retries} upload attempts failed")
                    return False
                    
            except Exception as e:
                print(f"[ERROR] Unexpected error during upload: {e}")
                return False
        
        return False
    
    def create_bucket_if_not_exists(self, bucket: str) -> bool:
        """
        Create S3 bucket if it doesn't exist.
        
        Args:
            bucket: Bucket name
            
        Returns:
            True if bucket exists or was created, False otherwise
        """
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=bucket)
            print(f"[✓] Bucket {bucket} already exists")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if self.region == 'us-east-1':
                        # us-east-1 doesn't need LocationConstraint
                        self.s3_client.create_bucket(Bucket=bucket)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=bucket,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    print(f"[✓] Created bucket {bucket} in region {self.region}")
                    return True
                    
                except ClientError as create_error:
                    print(f"[ERROR] Failed to create bucket {bucket}: {create_error}")
                    return False
            else:
                print(f"[ERROR] Error checking bucket {bucket}: {e}")
                return False
    
    def upload_directory(
        self, 
        local_dir: str, 
        bucket: str, 
        s3_prefix: str = "",
        exclude_patterns: Optional[list] = None
    ) -> bool:
        """
        Upload an entire directory to S3.
        
        Args:
            local_dir: Local directory path
            bucket: S3 bucket name
            s3_prefix: S3 prefix for uploaded files
            exclude_patterns: List of patterns to exclude
            
        Returns:
            True if all files uploaded successfully, False otherwise
        """
        if not os.path.exists(local_dir):
            print(f"[ERROR] Directory not found: {local_dir}")
            return False
        
        exclude_patterns = exclude_patterns or []
        success_count = 0
        total_count = 0
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                # Check exclude patterns
                if any(pattern in file for pattern in exclude_patterns):
                    continue
                
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = os.path.join(s3_prefix, relative_path).replace('\\', '/')
                
                total_count += 1
                if self.upload_file(local_path, bucket, s3_key):
                    success_count += 1
        
        print(f"[*] Upload summary: {success_count}/{total_count} files uploaded successfully")
        return success_count == total_count


def main():
    parser = argparse.ArgumentParser(description="Upload files to S3")
    parser.add_argument("local_path", help="Local file or directory path")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("s3_key", help="S3 key/prefix")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--content-type", help="Content type for file")
    parser.add_argument("--create-bucket", action="store_true", help="Create bucket if it doesn't exist")
    parser.add_argument("--exclude", nargs="*", help="Patterns to exclude when uploading directory")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    
    args = parser.parse_args()
    
    uploader = S3Uploader(region_name=args.region)
    
    # Create bucket if requested
    if args.create_bucket:
        if not uploader.create_bucket_if_not_exists(args.bucket):
            print("[ERROR] Failed to create/verify bucket")
            sys.exit(1)
    
    # Upload file or directory
    if os.path.isfile(args.local_path):
        success = uploader.upload_file(
            args.local_path, 
            args.bucket, 
            args.s3_key,
            max_retries=args.max_retries,
            content_type=args.content_type
        )
    elif os.path.isdir(args.local_path):
        success = uploader.upload_directory(
            args.local_path,
            args.bucket,
            args.s3_key,
            exclude_patterns=args.exclude or []
        )
    else:
        print(f"[ERROR] Path not found: {args.local_path}")
        sys.exit(1)
    
    if success:
        print("[✓] Upload completed successfully")
    else:
        print("[ERROR] Upload failed")
        sys.exit(1)


if __name__ == "__main__":
    main()