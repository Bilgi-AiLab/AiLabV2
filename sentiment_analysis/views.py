import json
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from sentiment_analysis.models import Report
from sentiment_analysis.algorithms.vader_web import vader
from project.models import Project
from sentiment_analysis.algorithms.textblob_web import textblob
from sentiment_analysis.algorithms.distilbert_web import distilbert
from sentiment_analysis.algorithms.roberta_web import roberta
from sentiment_analysis.algorithms.berturk_web import berturk
import pandas as pd
from io import BytesIO

def sentiment_algorithms(request, pk):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project=project)

    content = {'project': project, 'reports': reports, 'title': f'Sentiment Analysis - {project.title}'}

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Sentiment Analysis": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'sentiment_analysis/index.html', content)

def apply_sentiment_algorithm(request, pk, algorithm):
    project = get_object_or_404(Project, pk=pk)
    reports = Report.objects.filter(project_id=pk, algorithm=algorithm.lower())

    content = {'project': project, 'algorithm': algorithm, 'reports': reports,
               'title': f'{algorithm.upper()} - {project.title}'}

    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Sentiment Analysis": reverse('sentiment_algorithms', args=[pk]),
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
        if algorithm.lower() == "vader":
            output = vader(corpus=corpus)

        elif algorithm.lower() == "textblob":
            output = textblob(corpus=corpus)

        elif algorithm.lower() == "distilbert":
            output = distilbert(corpus=corpus)

        elif algorithm.lower() == "roberta":
            output = roberta(corpus=corpus)
        
        elif algorithm.lower() == "berturk":
            output = berturk(corpus=corpus)

        content.update(output)
        content["files"] = [file.filename() for file in files]

        def my_converter(o):
            return o.__str__()
        
        report = Report()
        report.project = project
        report.algorithm = algorithm.lower()
        report.all_data = json.dumps(output, separators=(',', ':'), default=my_converter)
        report.save()

        return redirect('view_sentiment_report', project.id, algorithm, report.id)
    
    return render(request, 'sentiment_analysis/params.html', content)

def view_sentiment_report(request, project_pk, algorithm, report_pk):
    project = get_object_or_404(Project, pk=project_pk)
    report = get_object_or_404(Report, pk=report_pk, algorithm=algorithm.lower())
    files = project.get_files()

    # Extract data from the report
    report_output = report.get_output()
    positive_count = report.positive_doc_count()
    negative_count = report.negative_doc_count()
    neutral_count = report.neutral_doc_count()

    # Create a bar chart using Plotly
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_count, negative_count, neutral_count]

    bar_chart = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=values,
                textposition='auto',
                marker=dict(color=['#28a745', '#dc3545', '#ffc107'])  # Green, Red, Yellow
            )
        ]
    )
    bar_chart.update_layout(
        title="Sentiment Analysis Results",
        xaxis_title="Sentiment",
        yaxis_title="Document Count",
        template="plotly_white"
    )

    # Convert Bar Chart to JSON
    bar_chart_json = json.dumps(bar_chart, cls=PlotlyJSONEncoder)

    # Create a pie chart
    pie_chart = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo='percent+label',
                marker=dict(colors=['#28a745', '#dc3545', '#ffc107'])  # Green, Red, Yellow
            )
        ]
    )
    pie_chart.update_layout(title="Sentiment Distribution")

    # Convert Pie Chart to JSON
    pie_chart_json = json.dumps(pie_chart, cls=PlotlyJSONEncoder)

    # Add data to the content dictionary
    content = {
        'project': project,
        'algorithm': algorithm,
        'files': [file.filename() for file in files],
        'report': report,
        'title': f'{algorithm.upper()} Report - {project.title}',
        'bar_chart': bar_chart_json,  # Add bar chart JSON
        'pie_chart': pie_chart_json  # Add pie chart JSON
    }

    # Add the report output to the content
    content.update(report_output)

    # Breadcrumb navigation
    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: reverse('show_project', args=[project.id]),
        "Sentiment Analysis": reverse('sentiment_algorithms', args=[project_pk]),
        algorithm.upper(): reverse('apply_sentiment_algorithm', args=[project_pk, algorithm]),
        f"Report (id:{report.id})": ""
    }

    content['breadcrumb'] = breadcrumb

    return render(request, 'sentiment_analysis/report.html', content)

def remove_sentiment_report(request, project_pk, algorithm, report_pk):
    report = get_object_or_404(Report, pk=report_pk)
    report.delete()

    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))

def download_excel(request, project_pk, algorithm, report_pk):
    project = get_object_or_404(Project, pk=project_pk)
    report = get_object_or_404(Report, pk=report_pk, algorithm=algorithm.lower())
    detailed_scores = report.detailed_scores()

    # Ensure detailed_scores is valid
    if not detailed_scores:
        return HttpResponse("No data available", content_type="text/plain")

    # Convert JSON string to a list if necessary
    if isinstance(detailed_scores, str):
        try:
            detailed_scores = json.loads(detailed_scores)
        except json.JSONDecodeError:
            return HttpResponse("Invalid JSON format", content_type="text/plain")

    # Convert the detailed scores to a DataFrame
    df = pd.DataFrame(detailed_scores)

    # Create an Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Detailed Scores")

    output.seek(0)

    # Create HTTP response
    response = HttpResponse(output.read(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = 'attachment; filename="detailed_scores.xlsx"'
    
    return response
    