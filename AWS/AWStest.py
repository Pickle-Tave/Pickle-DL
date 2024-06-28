from jiheonkey import *
import boto3
import requests
import os
import shutil

# s3 = boto3.client(
#     's3',
#     aws_access_key_id=AWS_S3_ACCESS_KEY,
#     aws_secret_access_key=AWS_S3_SECRET_KEY,
#     region_name='ap-northeast-2'
# )
# bucket_name = 's3-jiheon'

# memberid = ... # 유저 고유번호

# #단일 파일 실험, 다중 파일로 확장
# file_name = '20230813_172117.jpg'
# s3_key = memberid + '/' + file_name

# # 폴더 생성
# user_folder = memberid
# if not os.path.exists(user_folder):
#     os.makedirs(user_folder)
# local_file_path = os.path.join(user_folder, file_name)


# # 파일 다운로드
# s3.download_file(bucket_name, s3_key, local_file_path)
# print(f"File {file_name} downloaded to {local_file_path}")


# '''
# 모델에 다운받은 파일로 추론
# '''

# # 폴더 삭제
# shutil.rmtree(user_folder)
# print(f"Folder {user_folder} has been deleted")

# presigned URL로 이미지 다운로드 함수
def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Image downloaded successfully and saved to {save_path}")
    else:
        print("Error: Failed to download the image")


presigned_urls=["https://s3-jiheon.s3.ap-northeast-2.amazonaws.com/user123/20230813_172117.jpg?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEF0aDmFwLW5vcnRoZWFzdC0yIkcwRQIhANvycjwzimtjLf3ioVdLi%2F7W88fAtGegVmqAv5U1wrWqAiAIpUoVFrSrR3tzbRfu9%2BO0N%2B5hKtZwvzlCB3Shcp1L2iroAggWEAAaDDkwNTQxODM5NzY4NSIM56nhc0LGPuHqs%2BWLKsUCVH3wB6MTjwMhg7jd0qfn6DvF9K6Y0y9PJHDSAC37D679MjgMkQ9bIyvKuvih6TdRbwZY%2BvJh8%2B%2Bk1Ggc2HT4YA2MySTUuPfc3HCblrCFoce20YRyYh7L4dXLc8oEWRnRJg4nDSDPNcC80439iEqJOwojC3pMgbxjt3J2AtB08usEkQRwuRFghII38%2BFP%2BBaLfvCsY7Cx%2BLGHHSHgnh%2FID%2BzZrvjpb8zjx4w5sPxxbWn3%2FdzIBKqodnPxaLL4D5YtIgEn4vQgMwknI1FDT%2BXjyAKT1kVGUFhs3goIsNNvI%2B%2B44Q2S3YW%2BJttSPLXGTICFnckvzp84iUT5KJNmMiouer3GAIpNIMUJOZrRpTgkRquWCSLNOwJiubAhFGQn446uz0nnApwTrwO3zBOAhk9qwz3iQaGztq49tXpWl9P8CYnMYJAmnTDQ4fqzBjqzAsbkozqLE8AO65Yk0ewGvJG44O%2FTppILBywUheHqdyKAzuoYa0SWerxTbsE7SFpzzf00gqtpDafiqqDOt1sBGXE88tobrGQHqlZdsEk5S8YkqVodS9swfqgtnUvATTNu5kKbwI878g3xJxXYjdytxulfiPqcQhgUqatkK0a6y0%2FjPEtN6QJinK1WvX3BimimyGZxcqPYmzZr1urwfB8SCGLNrx5RW%2FCW9%2FOKofyUqbggszfIBJfHnzKt7f18oNs2%2BZY7EaMoaAIM2ynOOZ72z8%2FChm9ZXGVll5uZTaV06zz5hSzfVQ0ZM0P8UfRQ7WhTBv4uW57GqbIHZ84gbYGs0E80bPT%2FVv4XxSMqbN8fafqIs44fFe2zTZVaUEriqToXiZEqPa%2FkgE9585YRH8yWP5zYgS8%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240628T125400Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=ASIA5FTZEK72R25GV6MV%2F20240628%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Signature=45d5bb437667bc7608501e3f4112218b57d5b985e0b14fd35a3b2b1fbe200ebd",
"https://s3-jiheon.s3.ap-northeast-2.amazonaws.com/user123/image.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEF0aDmFwLW5vcnRoZWFzdC0yIkcwRQIhANvycjwzimtjLf3ioVdLi%2F7W88fAtGegVmqAv5U1wrWqAiAIpUoVFrSrR3tzbRfu9%2BO0N%2B5hKtZwvzlCB3Shcp1L2iroAggWEAAaDDkwNTQxODM5NzY4NSIM56nhc0LGPuHqs%2BWLKsUCVH3wB6MTjwMhg7jd0qfn6DvF9K6Y0y9PJHDSAC37D679MjgMkQ9bIyvKuvih6TdRbwZY%2BvJh8%2B%2Bk1Ggc2HT4YA2MySTUuPfc3HCblrCFoce20YRyYh7L4dXLc8oEWRnRJg4nDSDPNcC80439iEqJOwojC3pMgbxjt3J2AtB08usEkQRwuRFghII38%2BFP%2BBaLfvCsY7Cx%2BLGHHSHgnh%2FID%2BzZrvjpb8zjx4w5sPxxbWn3%2FdzIBKqodnPxaLL4D5YtIgEn4vQgMwknI1FDT%2BXjyAKT1kVGUFhs3goIsNNvI%2B%2B44Q2S3YW%2BJttSPLXGTICFnckvzp84iUT5KJNmMiouer3GAIpNIMUJOZrRpTgkRquWCSLNOwJiubAhFGQn446uz0nnApwTrwO3zBOAhk9qwz3iQaGztq49tXpWl9P8CYnMYJAmnTDQ4fqzBjqzAsbkozqLE8AO65Yk0ewGvJG44O%2FTppILBywUheHqdyKAzuoYa0SWerxTbsE7SFpzzf00gqtpDafiqqDOt1sBGXE88tobrGQHqlZdsEk5S8YkqVodS9swfqgtnUvATTNu5kKbwI878g3xJxXYjdytxulfiPqcQhgUqatkK0a6y0%2FjPEtN6QJinK1WvX3BimimyGZxcqPYmzZr1urwfB8SCGLNrx5RW%2FCW9%2FOKofyUqbggszfIBJfHnzKt7f18oNs2%2BZY7EaMoaAIM2ynOOZ72z8%2FChm9ZXGVll5uZTaV06zz5hSzfVQ0ZM0P8UfRQ7WhTBv4uW57GqbIHZ84gbYGs0E80bPT%2FVv4XxSMqbN8fafqIs44fFe2zTZVaUEriqToXiZEqPa%2FkgE9585YRH8yWP5zYgS8%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240628T125435Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=ASIA5FTZEK72R25GV6MV%2F20240628%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Signature=f536599bd4408cc87963e0514c60ec464f67093c4818075860378397f5bd9dd8",
"https://s3-jiheon.s3.ap-northeast-2.amazonaws.com/user123/%E1%84%81%E1%85%A1%E1%86%B7%E1%84%8C%E1%85%A1_%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%86%E1%85%A5.jpeg?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEF0aDmFwLW5vcnRoZWFzdC0yIkcwRQIhANvycjwzimtjLf3ioVdLi%2F7W88fAtGegVmqAv5U1wrWqAiAIpUoVFrSrR3tzbRfu9%2BO0N%2B5hKtZwvzlCB3Shcp1L2iroAggWEAAaDDkwNTQxODM5NzY4NSIM56nhc0LGPuHqs%2BWLKsUCVH3wB6MTjwMhg7jd0qfn6DvF9K6Y0y9PJHDSAC37D679MjgMkQ9bIyvKuvih6TdRbwZY%2BvJh8%2B%2Bk1Ggc2HT4YA2MySTUuPfc3HCblrCFoce20YRyYh7L4dXLc8oEWRnRJg4nDSDPNcC80439iEqJOwojC3pMgbxjt3J2AtB08usEkQRwuRFghII38%2BFP%2BBaLfvCsY7Cx%2BLGHHSHgnh%2FID%2BzZrvjpb8zjx4w5sPxxbWn3%2FdzIBKqodnPxaLL4D5YtIgEn4vQgMwknI1FDT%2BXjyAKT1kVGUFhs3goIsNNvI%2B%2B44Q2S3YW%2BJttSPLXGTICFnckvzp84iUT5KJNmMiouer3GAIpNIMUJOZrRpTgkRquWCSLNOwJiubAhFGQn446uz0nnApwTrwO3zBOAhk9qwz3iQaGztq49tXpWl9P8CYnMYJAmnTDQ4fqzBjqzAsbkozqLE8AO65Yk0ewGvJG44O%2FTppILBywUheHqdyKAzuoYa0SWerxTbsE7SFpzzf00gqtpDafiqqDOt1sBGXE88tobrGQHqlZdsEk5S8YkqVodS9swfqgtnUvATTNu5kKbwI878g3xJxXYjdytxulfiPqcQhgUqatkK0a6y0%2FjPEtN6QJinK1WvX3BimimyGZxcqPYmzZr1urwfB8SCGLNrx5RW%2FCW9%2FOKofyUqbggszfIBJfHnzKt7f18oNs2%2BZY7EaMoaAIM2ynOOZ72z8%2FChm9ZXGVll5uZTaV06zz5hSzfVQ0ZM0P8UfRQ7WhTBv4uW57GqbIHZ84gbYGs0E80bPT%2FVv4XxSMqbN8fafqIs44fFe2zTZVaUEriqToXiZEqPa%2FkgE9585YRH8yWP5zYgS8%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240628T125455Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=ASIA5FTZEK72R25GV6MV%2F20240628%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Signature=d6bb8ebf29de318847987816f7e598230a7f88eeca6eb6670e6ed301e8ef6bae"]

# presigned URL 생성


for idx, presigned_url in enumerate(presigned_urls):
    if presigned_url:
        # 이미지 다운로드 및 저장 경로
        save_path = presigned_url[:20] + '.jpg'
        download_image(presigned_url, save_path)
    else:
        print("Failed to generate presigned URL.")
