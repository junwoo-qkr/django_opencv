from django.contrib import admin
from .models import ImageUploadModel

# Register your models here.
class ImageUploadAdmin(admin.ModelAdmin):
    list_display = ('description', 'document') # list_display 변수명은 고정
admin.site.register(ImageUploadModel, ImageUploadAdmin)