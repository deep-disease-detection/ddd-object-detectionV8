from google.cloud import storage
import time
import os


def save_image_to_bucket_gcp(image, bucket, virus, virus_count):
    """Save image to bucket in GCP
    Args:
         image (bytes): Image to be saved
         bucket (str): Bucket name
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)
    name_photo = str(virus)+".jpg"
    # add timestamp to name
    name_photo = name_photo.split(".")[0] + "-" + str(time.time()) + ".jpg"
    blob = bucket.blob(os.path.basename(name_photo))
    # add metadata to image
    blob.metadata = {"time": str(time.time()), "class": virus, "virus_count":virus_count}
    blob.upload_from_string(image, content_type="image/jpeg")
    print("Image saved to bucket")
