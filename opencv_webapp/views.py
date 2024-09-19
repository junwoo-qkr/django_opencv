from django.shortcuts import render
from .forms import SimpleUploadForm, ImageUploadForm
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .cv_functions import cv_detect_face
# Create your views here.

def first_view(request):
    return render(request, 'opencv_webapp/first_view.html', {})


def simple_upload(request):
    if request.method == 'POST':
        # request.POST : title
        # request.FILES : image
        form = SimpleUploadForm(request.POST, request.FILES)
        if form.is_valid():
            myfile = request.FILES['image'] # 업로드된 이미지
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile) # myfile.name = 저장이 끝난 물리적인 파일의 이름
            uploaded_file_url = fs.url(filename) # myfile로 접근 가능한 url
            context = {'form': form, 'uploaded_file_url': uploaded_file_url}
            return render(request, 'opencv_webapp/simple_upload.html', context)
    else:
        form = SimpleUploadForm()
        context = {'form':form}
        return render(request, 'opencv_webapp/simple_upload.html', context)
    

def detect_face(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False) # 임시저장한 값을 post에 저장(후처리작업을 위해)
            post.save()
            # document : ImageUploadModel Class에 선언되어 있는 “document”에 해당
            imageURL = settings.MEDIA_URL + form.instance.document.name
            # == form.instance.document.url
            # == post.document.url
            # == '/media/images/2021/10/29/ses_XQAftn4.jpg'
            # print(form.instance, form.instance.document.name, form.instance.document.url)
            cv_detect_face(settings.MEDIA_ROOT_URL + imageURL) # 추후 구현 예정
            return render(request, 'opencv_webapp/detect_face.html', {'form':form, 'post':post})
    else:
        form = ImageUploadForm()
        return render(request, 'opencv_webapp/detect_face.html', {'form': form})