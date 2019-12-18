from django.shortcuts import render
from django.http import HttpResponse
import json
def index(request):
    return render(request,'index.html')

def getSummary(request):
    data=json.loads(request.body)
    print(data)
    #todo
    # summary_array=[]
    # summary1=tf_idf_summary(data['content'])
    # summary_array.append(summary1)
    # summary2=textrank_summary()


    res={
        "summary":[
            'summary1',
            'summary2',
            'summary3'
        ],
        "image":[
            "/static/1.jpg",
            "/static/1.jpg"

        ]
    }
    ####
    return HttpResponse(json.dumps(res),content_type="application/json")