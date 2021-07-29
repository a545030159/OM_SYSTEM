import sys
sys.path.extend(["../","./"])
from django.http import HttpResponse
from django.shortcuts import render
# from mining.entry import mining
# from mining.whole_model import mine_model
from mining.whole_model import decode
from mining.whole_model import load_model
from mining.whole_model import crawler_pro
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.clickjacking import xframe_options_sameorigin
import json
import os


path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input.txt')
path_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input_1.txt')


def index1(request):
    return render(request, "index1.html", {})


def index2(request):
    return render(request, "index2.html", {})


# def index(request):
#     return render(request, "index.html", {})


def mine(request):
    return HttpResponse(decode('input.txt'))


def mine_1(request):
    return HttpResponse(decode('input_1.txt'))


def load(request):
    try:
        load_model()
    except Exception:
        return HttpResponse(json.dumps({'result': Exception}))
    else:
        return HttpResponse(json.dumps({'result': 'success'}))


@csrf_exempt
def crawlers(request):
    try:
        v1 = request.POST.get('v1')
        v2 = request.POST.get('v2')
        id = request.POST.get('id')
        return_text = crawler_pro(v1, v2,id)
    except Exception:
        return HttpResponse(json.dumps({'result': Exception}))
    else:
        return HttpResponse(return_text)



@csrf_exempt
def upload(request):
    returnValue= ''
    num = 0
    if request.method == 'POST':
        files = request.FILES.getlist('fileToUpload', None)
        with open(path, 'w', encoding='utf8') as f:
            for file in files:
                for line in file.readlines():
                    a = bytes.decode(line)
                    a = a.replace('\r', '')
                    num += 1
                    f.write(a)
                    num_string = str(num)
                    num_string += ' '
                    returnValue += (num_string + a)
    return HttpResponse(returnValue)


@csrf_exempt
def upload_1(request):
    returnValue= ''
    num = 0
    if request.method == 'POST':
        files = request.FILES.getlist('fileToUpload', None)
        with open(path_1, 'w', encoding='utf8') as f:
            for file in files:
                for line in file.readlines():
                    a = bytes.decode(line)
                    a = a.replace('\r', '')
                    num += 1
                    f.write(a)
                    num_string = str(num)
                    num_string += ' '
                    returnValue += (num_string + a )
    return HttpResponse(returnValue)


@csrf_exempt
@xframe_options_sameorigin
def post(request):
    returnValue = ''
    num = 0
    if request.method == 'POST':
        textarea = request.POST.get('textarea')
        textLst = textarea.strip('\n').split('\n')
        with open(path, 'w', encoding='utf8') as f:
            for text in textLst:
                text = text.replace('\r', '')
                num += 1
                f.write(text+'\n')
                num_string = str(num)
                num_string += ' '
                returnValue += (num_string + text + '\n')
    return HttpResponse(returnValue)

@csrf_exempt
@xframe_options_sameorigin
def post_1(request):
    returnValue = ''
    num = 0
    if request.method == 'POST':
        textarea = request.POST.get('textarea')
        textLst = textarea.strip('\n').split('\n')
        with open(path_1, 'w', encoding='utf8') as f:
            for text in textLst:
                text = text.replace('\r', '')
                num += 1
                f.write(text+'\n')
                num_string = str(num)
                num_string += ' '
                returnValue += (num_string + text + '\n')
    return HttpResponse(returnValue)