from django.contrib import admin

# Register your models here.

from .models import Post

# 관리자(admin)가 게시글(Post)에 접근 가능
admin.site.register(Post)