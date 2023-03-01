from contextlib import nullcontext
from django.db import models
from django.core.files.uploadedfile import InMemoryUploadedFile
import sys
from PIL import Image
import io
import torch
from torchvision import transforms as T
# Create your models here.

class Post(models.Model):

    postname = models.CharField(max_length=50)
    contents = models.TextField(null=True)

    # 게시글 Post에 이미지 추가
    mainphoto = models.ImageField(upload_to = "Uploaded Files/", blank=True, null=True)

    # 게시글의 제목(postname)이 Post object 대신하기
    def __str__(self):
        return self.postname

class Document(models.Model):
    # 업로드 된 파일의 사용자 지정 제목을 저장하기위한 문자 유형
    title = models.CharField(max_length = 200)
    # 파일을 저장하기 위해FileField()를 사용
    uploadedFile = models.FileField(upload_to = "Uploaded_Files/")
    # 파일 업로드 날짜 및 시간을 저장
    dateTimeOfUpload = models.DateTimeField(auto_now = True)

    image_converted = models.ImageField(upload_to = "result_files/", null=True)


