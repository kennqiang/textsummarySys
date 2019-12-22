from django.shortcuts import render
from django.http import HttpResponse
import json
import myapp.tf_idf_summary as my_summary1
from myapp.TextRank import TextRank
from myapp.seq2seq import seq2seq

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

    ob=TextRank(data['content'])
    summary2=ob.best(1)
    # summary_array.append(summary1)
    # summary2=textrank_summary()

    ob2=seq2seq()
    summary3=ob2.predict(data['content'])

    res={
        "summary":[
            summary1,
            "tf-idf score："+str(score),
            summary2,
            # data['content'],
            summary3
            # 'summary3'
        ],
        "image":[
            # "/static/1.jpg",
            # "/static/1.jpg",

        ]
    }
    ####
    return HttpResponse(json.dumps(res),content_type="application/json")