import json
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from summarization.models import Report
from project.models import Project
from summarization.algorithms.pegasus_web import pegasus
from summarization.algorithms.bart_web import bart
from summarization.algorithms.t5_web import t5
from summarization.algorithms.brio_web import brio
from summarization.algorithms.deepseek_web import deepseek
from summarization.algorithms.lexrank_web import lexrank

def summarization_algorithms(request, pk):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project=project)

    content = {'project': project, 'reports': reports, 'title': f'Summarization - {project.title}'}

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Summarization": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'summarization/index.html', content)

def apply_summarization_algorithm(request, pk, algorithm):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project_id=pk, algorithm=algorithm.lower())

    content = {'project': project, 'algorithm': algorithm, 'reports': reports,
               'title': f'{algorithm.upper()} - {project.title}'}

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Summarization": reverse('summarization_algorithms', args=[pk]),
        algorithm.upper(): ""
        
    }

    content['breadcrumb'] = breadcrumb


    if request.method == 'POST':

        files = project.get_files()
        corpus = []
        
        for file in files:
            file = open(file.file.path, "r", encoding='utf8')
            lines = file.read()
            file.close()
            corpus.append(lines)
        
        output = {}
        if algorithm.lower() == "pegasus":
            output = pegasus(corpus)

        elif algorithm.lower() == "bart":
            output = bart(corpus)

        elif algorithm.lower() == "t5":
            output = t5(corpus)

        elif algorithm.lower() == "brio":
            output = brio(corpus)

        elif algorithm.lower() == "deepseek":
            output = deepseek(corpus)

        elif algorithm.lower() == "lexrank":
            output = lexrank(corpus)


        content.update(output)
        content["files"] = [file.filename() for file in files]

        def my_converter(o):
            return o.__str__()
        
        report = Report()
        report.project = project
        report.algorithm = algorithm.lower()
        report.all_data = json.dumps(output, separators=(',', ':'), default=my_converter)
        report.save()

        return redirect('view_summarization_report', project.id, algorithm, report.id)
    
    return render(request, 'summarization/params.html', content)

def view_summarization_report(request, project_pk, algorithm, report_pk):
    project = get_object_or_404(Project, pk=project_pk)
    report = get_object_or_404(Report, pk=report_pk, algorithm=algorithm.lower())
    files = project.get_files()

    # Extract data from the report
    report_output = report.get_output()
    summary = report.summary()
    rouge1 = report.rouge1()
    rouge2 = report.rouge2()
    rougeL = report.rougeL()

    rouge1_precision = rouge1[0]
    rouge1_recall = rouge1[1]
    rouge1_f1 = rouge1[2]

    rouge2_precision = rouge2[0]
    rouge2_recall = rouge2[1]
    rouge2_f1 = rouge2[2]

    rougeL_precision = rougeL[0]
    rougeL_recall = rougeL[1]
    rougeL_f1 = rougeL[2]

    bert_score = report.bert_score()
    # Add data to the content dictionary
    content = {
        'project': project,
        'algorithm': algorithm,
        'files': [file.filename() for file in files],
        'summary': summary,
        'rouge1_precision': rouge1_precision,
        'rouge1_recall': rouge1_recall,
        'rouge1_f1': rouge1_f1,
        'rouge2_precision': rouge2_precision,
        'rouge2_recall': rouge2_recall,
        'rouge2_f1': rouge2_f1,
        'rougeL_precision': rougeL_precision,
        'rougeL_recall': rougeL_recall,
        'rougeL_f1': rougeL_f1,
        'bert_score': bert_score,
        'report': report,
        'title': f'{algorithm.upper()} Report - {project.title}',
    }

    # Add the report output to the content
    content.update(report_output)

    # Breadcrumb navigation
    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Summarization": reverse('summarization_algorithms', args=[project_pk]),
        algorithm.upper(): reverse('apply_summarization_algorithm', args=[project_pk, algorithm]),
        f"Report (id:{report.id})": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'summarization/report.html', content)

def remove_summarization_report(request, project_pk, algorithm, report_pk):
    report = get_object_or_404(Report, pk=report_pk)
    report.delete()

    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))