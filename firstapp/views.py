from multiprocessing import context
from django.shortcuts import render, redirect

from django.core.files.storage import FileSystemStorage
from django.core.files.uploadedfile import InMemoryUploadedFile

from .models import Post, Document


from types import SimpleNamespace
from torchvision import transforms as T
import torch
from PIL import Image
import io
from torchvision.utils import save_image
import os
import time

from .StarGAN.solver import Solver
from keras.preprocessing import image
from .StarGAN.model import *

from .StarGAN2.solver import Solver as Solver2
from .StarGAN2.model import *
from .StarGAN2.data_loader import get_test_loader

StarGAN_config = SimpleNamespace()
StarGAN2_args = SimpleNamespace()

# Training configuration.
StarGAN_config.dataset = "CelebA"
StarGAN_config.selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
StarGAN_config.g_lr = 0.0001
StarGAN_config.d_lr = 0.0001
StarGAN_config.beta1 = 0.5
StarGAN_config.beta2 = 0.999
StarGAN_config.batch_size = 16
StarGAN_config.num_iters = 200000
StarGAN_config.num_iters_decay = 100000
StarGAN_config.n_critic = 5
StarGAN_config.resume_iters = None

# Test configuration.
StarGAN_config.test_iters = 200000

# Model configurations.
StarGAN_config.c_dim = 5
StarGAN_config.c2_dim = 8
StarGAN_config.image_size = 128
StarGAN_config.g_conv_dim = 64
StarGAN_config.d_conv_dim = 64
StarGAN_config.g_repeat_num = 6
StarGAN_config.d_repeat_num = 6
StarGAN_config.lambda_cls = 1
StarGAN_config.lambda_rec = 10
StarGAN_config.lambda_gp = 10

# Directories.
StarGAN_config.model_save_dir = "firstapp/StarGAN/models"

# Miscellaneous.
StarGAN_config.num_workers = 1
StarGAN_config.mode = "test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################################
StarGAN2_args.img_size = 256
StarGAN2_args.num_domains = 2
StarGAN2_args.mode = 'sample'
StarGAN2_args.latent_dim = 16
StarGAN2_args.hidden_dim = 512
StarGAN2_args.style_dim = 64
StarGAN2_args.val_batch_size = 8
StarGAN2_args.lambda_reg = 1
StarGAN2_args.lambda_cyc = 1
StarGAN2_args.lambda_sty = 1
StarGAN2_args.lambda_ds = 1
StarGAN2_args.ds_iter = 100000
StarGAN2_args.resume_iter = 100000
StarGAN2_args.w_hpf = 1

StarGAN2_args.num_workers = 0
StarGAN2_args.seed = 777

StarGAN2_args.checkpoint_dir = 'firstapp\\StarGAN2\\checkpoints\\celeba_hq'
StarGAN2_args.result_dir = "media\\Uploaded_Files"
StarGAN2_args.src_dir = "media\\Uploaded_Files\\celeba_hq\\src"
StarGAN2_args.ref_dir = "media\\Uploaded_Files\\celeba_hq\\ref"
StarGAN2_args.wing_path = "firstapp\\StarGAN2\\checkpoints\\wing.ckpt"
# StarGAN2_args.lm_path = ""

##################################

StarGAN_solver = Solver(StarGAN_config)
StarGAN_solver.restore_model(200000) # 200000 iterations

StarGAN2_solver = Solver2(StarGAN2_args)


# index.html 페이지를 부르는 index 함수
def index(request):
    return render(request,'firstapp/index.html')

# blog.html 페이지를 부르는 blog 함수
def blog(request):
	# 모든 Post를 가져와 postlist에 저장합니다
    postlist = Post.objects.all()
	# blog.html 페이지를 열 때, 모든 Post인 postlist도 같이 가져옵니다
    return render(request, 'firstapp/blog.html', {'postlist':postlist})


# blog의 게시글(posting)을 부르는 posting 함수
def posting(request, pk):
    # 게시글(Post) 중 pk(primary_key)를 이용해 하나의 게시글(post)를 검색
    post = Post.objects.get(pk=pk)
    # posting.html 페이지를 열 때, 찾아낸 게시글(post)을 post라는 이름으로 가져옴
    return render(request, 'firstapp/posting.html', {'post':post})

def remove_post(request, pk):
    post = Post.objects.get(pk=pk)
    if request.method == 'POST':
        post.delete()
        return redirect('/blog/')
    return render(request, 'firstapp/remove_post.html', {'Post': post})

def uploadFile(request):
    global document
    if request.method == "POST":
        # Fetching the form data
        fileTitle = request.POST["fileTitle"]
        uploadedFile = request.FILES["uploadedFile"]

        # Saving the information in the database
        document = Document(
            title = fileTitle,
            uploadedFile = uploadedFile
        )
        document.save()

        documents = Document.objects.all()
        context = {"files": documents,
                    "result":document }
    else:
        documents = Document.objects.all()
        context = {"files": documents}

    return render(request, "firstapp/upload-file.html", context)

def StarGAN(request):
    fileObj = request.FILES['uploadedFile']
    select_origin_hair = request.POST.getlist('OriginHairRadio')
    select_origin_gender = request.POST.getlist('OriginGenderRadio')
    select_origin_age = request.POST.getlist('OriginAgeRadio')
    c_org=[[0,0,0,0,0]]
    if select_origin_hair[0]=='blond':
        c_org[0][1]=1
    elif select_origin_hair[0]=='black':
        c_org[0][0]=1
    else:
        c_org[0][2]=1
    if select_origin_gender[0]=='male':
        c_org[0][3]=1
    else:
        c_org[0][3]=0
    if select_origin_age[0]=='young':
        c_org[0][4]=1
    else:
        c_org[0][4]=0
    select_hair = request.POST.getlist('HairRadio')
    select_gender = request.POST.getlist('GenderRadio')
    select_age = request.POST.getlist('AgeRadio')
    c_trg=c_org
    c_org = torch.FloatTensor(c_org).to(device)
    if select_hair[0]=='blond':
        c_trg[0][1]=1
        c_trg[0][0]=0
        c_trg[0][2]=0
    elif select_hair[0]=='black':
        c_trg[0][0]=1
        c_trg[0][1]=0
        c_trg[0][2]=0
    else:
        c_trg[0][2]=1
        c_trg[0][0]=0
        c_trg[0][1]=0
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(178,218))
    transform = []
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    img = transform(img).unsqueeze(0).to(device)
    result_origin_img = img
    result_origin_img = StarGAN_solver.denorm(result_origin_img.data.cpu()).squeeze(0)
    save_image(result_origin_img, os.path.join(os.getcwd(), 'media/Uploaded_Files/test.jpg'))
    result_origin_img = 'Uploaded_Files/test.jpg'
    document = Document(title = "result", uploadedFile=result_origin_img)
    document.save()
#########################################################################
    c_trg = torch.FloatTensor(c_trg).to(device)
    result1_img = img
    _, _, _, output = StarGAN_solver.model(result1_img, c_trg, c_org)
    result1_img = StarGAN_solver.denorm(output.data.cpu()).squeeze(0)
    save_image(result1_img, os.path.join(os.getcwd(), 'media/Uploaded_Files/result1.jpg'))
    result1_img = 'Uploaded_Files/result1.jpg'
    document1 = Document(title = "result1", uploadedFile=result1_img)
    document1.save()
####################### 성별 적용#########################################
    if select_gender[0]=='male':
        c_trg[0][3]=1
    else:
        c_trg[0][3]=0
    c_trg = torch.FloatTensor(c_trg).to(device)
    result2_img = img
    _, _, _, output = StarGAN_solver.model(result2_img, c_trg, c_org)
    result2_img = StarGAN_solver.denorm(output.data.cpu()).squeeze(0)
    save_image(result2_img, os.path.join(os.getcwd(), 'media/Uploaded_Files/result2.jpg'))
    result2_img = 'Uploaded_Files/result2.jpg'
    document2 = Document(title = "result2", uploadedFile=result2_img)
    document2.save()
##################### 나이 적용 ################################
    if select_age[0]=='young':
        c_trg[0][4]=1
    else :
        c_trg[0][4]=0
    c_trg = torch.FloatTensor(c_trg).to(device)
    result3_img = img
    _, _, _, output = StarGAN_solver.model(result3_img, c_trg, c_org)
    result3_img = StarGAN_solver.denorm(output.data.cpu()).squeeze(0)
    save_image(result3_img, os.path.join(os.getcwd(), 'media/Uploaded_Files/result3.jpg'))
    result3_img = 'Uploaded_Files/result3.jpg'
    document3 = Document(title = "result3", uploadedFile=result3_img)
    document3.save()
########################################################
    documents = Document.objects.all()
    context = {"files": documents,
            "result":document,
            "result1":document1,
            "result2":document2,
            "result3":document3,
            "origin":filePathName }
    return render(request, "firstapp/upload-file.html", context)

def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]

def StarGAN2(request):

    assert len(subdirs(StarGAN2_args.src_dir)) == StarGAN2_args.num_domains
    assert len(subdirs(StarGAN2_args.ref_dir)) == StarGAN2_args.num_domains
    loaders = Munch(src=get_test_loader(root=StarGAN2_args.src_dir,
                                                img_size=StarGAN2_args.img_size,
                                                batch_size=StarGAN2_args.val_batch_size,
                                                shuffle=False,
                                                num_workers=StarGAN2_args.num_workers),
                            ref=get_test_loader(root=StarGAN2_args.ref_dir,
                                                img_size=StarGAN2_args.img_size,
                                                batch_size=StarGAN2_args.val_batch_size,
                                                shuffle=False,
                                                num_workers=StarGAN2_args.num_workers))
    StarGAN2_solver.sample(loaders)
    result_img = 'Uploaded_Files/reference.jpg'
    document = Document(title = "result", uploadedFile=result_img)
    context = {"result":document}
    return render(request, "firstapp/upload-file.html", context)