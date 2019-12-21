from django.shortcuts import render
from django.http import HttpResponse
import json
import myapp.tf_idf_summary as my_summary1
# from TextRank import TextRank

def index(request):
    return render(request,'index.html')

def getSummary(request):
    data=json.loads(request.body)
    print(data)
    #todo
    # summary_array=[]
    result=my_summary1.summary1(data['content'])
    summary1=result[0]
    score=result[1]
    # ob=TextRank(data['content'])
    # sresult2=ob.best(sentenceNum)
    # summary_array.append(summary1)
    # summary2=textrank_summary()

    res={
        "summary":[
            summary1,
            score,
            data['content'],
            'summary3'
        ],
        "image":[
            "/static/1.jpg",
            "/static/1.jpg",

        ]
    }
    ####
    return HttpResponse(json.dumps(res),content_type="application/json")