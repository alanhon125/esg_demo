from django.shortcuts import render
from django.http import HttpResponse
from django.db.models import Q
import json
from transformers import AutoTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
from collections import defaultdict
import random
random.seed(5)
from tqdm import tqdm
import sys
from argparse import ArgumentParser
sys.path.append('..')
import nvidia_smi


async def genius_inference(request):
    """

    """
    request = json.loads(request.body.decode("utf-8"))
    print(request)
    keywords = request.get("keywords")
    model_type = '1' #no-num-finetuned model
    #model_type = '2' #with-num-finetuned model
    aspects = request.get("aspect")
    try:
        num = int(request.get("number_of_result"))
    except:
        response = {
            "success": False,
            "message": "Please provide an integer within 1 and 5."
        }
        return HttpResponse(json.dumps(response, indent=4))
    #############################################################################################################################
    if num > 5 or num < 1:
        response = {
            "success": False,
            "message": "Please provide an integer within 1 and 5."
        }
        return HttpResponse(json.dumps(response, indent=4))

    Attributes = aspects#["Oil and gas industry: ","Coal industry: ","Electricity industry: ","Property development industry: "," "]
    #sketch = "<mask> supply chain <mask> climate change <mask> climate action <mask> renewable energy <mask> green finance <mask> "
    #sketch = "<mask> property development project <mask> harzardous waste <mask> non-hazardous waste <mask> recycle <mask> waste water <mask>"

    keywords_list = (i.strip() for i in keywords)
    sketch = '<mask> ' + ' <mask> '.join(keywords_list) + ' <mask> '
    if model_type == '1':
        ckpt = '/home/data1/public/ResearchHub/ESG_text_generation/models/checkpoint-no-num-970/'#/home/data/Research/ESG_text_generation/models/checkpoint-no-num-970
    else:
        ckpt = '/home/data1/public/ResearchHub/ESG_text_generation/models/checkpoint-with-num-2685/'
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    genius_model = BartForConditionalGeneration.from_pretrained(ckpt)
    #choose the gpu used for inference in case the gpu utilization < 40%
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f"|Device {i}| Mem Free: {mem.free/1024**2:5.2f}MB / {mem.total/1024**2:5.2f}MB | gpu-util: {util.gpu:3.1%} | gpu-mem: {util.memory:3.1%} |")
        print('gpu utilizagtion: ',util.gpu)
        if util.gpu < 40:
            genius_generator = Text2TextGenerationPipeline(genius_model, tokenizer, device=i)
            break
        if i==3:
            response = {
                        "success": False,
                        "message": "There is no enough gpu-util in this server."
                        }
            return HttpResponse(json.dumps(response, indent=4))
    #genius_generator

    print('--generating GENIUS output from sketches--')
    #print("###########################################INPUT###########################################\n",sketch)
    outputs = []
    if len(Attributes) > 0:
        for i in tqdm(range(len(Attributes))):
            attr_output = []
            for j in range(num):
                generation = genius_generator(Attributes[i] + ": " + sketch, max_length=2000, do_sample=True, num_beams=5)[0]['generated_text']
                attr_output.append(generation)
            outputs.append(attr_output)
    else:
        attr_output = []
        for j in range(num):
            generation = genius_generator(sketch, max_length=2000, do_sample=True, num_beams=5)[0]['generated_text']
            attr_output.append(generation)
        outputs.append(attr_output)
    #print("###########################################OUTPUT###########################################\n")
    #for i in outputs:
    #    print(i,'\n')
    #sketch
    output_dict = {"success": True,"outputs": outputs}

    return HttpResponse(json.dumps(output_dict))