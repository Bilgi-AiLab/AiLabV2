import os
import shutil
import zipfile
from django.core.files.base import ContentFile
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse

from project.helpers import pdf_to_text, excel_to_text
from project.models import Project, ProjectFile
import pandas as pd
import uuid

def all_projects(request):
    projects = Project.objects.all()
    content = {'projects': projects, 'title': 'Projects'}
    return render(request, "project/project-list.html", content)


def show_project(request, pk):
    project = get_object_or_404(Project, pk=pk)

    content = {'project': project, 'title': project.title}
    breadcrumb = {
        "Projects": reverse('all_projects'),
        project.title: "",
    }

    content['breadcrumb'] = breadcrumb
    return render(request, "project/project-show.html", content)


def create_project(request):
    if request.method == 'POST':
        author = request.user
        name = ""
        files = []
        if 'text' in request.POST:
            name = request.POST['name']
            text = request.POST['text']

            # Convert text input into an in-memory .txt file
            text_file = ContentFile(text.encode('utf-8'), name=f"{name}.txt")
            files.append(text_file)
        elif 'files' in request.FILES:
            name = request.POST['name']
            files = request.FILES.getlist('files')
        elif 'folder_files' in request.FILES:
            name = request.POST['folder_name']
            files = request.FILES.getlist('folder_files')
        
        project = Project(title=name,author=author)
        project.save()

        add_files_to_project(files, project)

        return redirect('all_projects')

    content = {'title': 'New Project'}
    breadcrumb = {
        "Projects": reverse('all_projects'),
        "Create Project": "",
    }
    content['breadcrumb'] = breadcrumb

    return render(request, "project/project-new.html", content)


def delete_project(request, pk):
    project = get_object_or_404(Project, pk=pk)
    project_folder = f'data/projects/{project.project_folder}'
    try:
        shutil.rmtree(project_folder)
    except FileNotFoundError:
        pass
    finally:
        pass
    project.delete()

    return redirect('all_projects')


def add_files(request, pk):
    if request.method == 'POST':
        project = get_object_or_404(Project, pk=pk)

        files = []
        if 'files' in request.FILES:
            files = request.FILES.getlist('files')
        elif 'folder_files' in request.FILES:
            files = request.FILES.getlist('folder_files')
            
        add_files_to_project(files, project)

        return redirect('show_project', pk=pk)
    return None


def add_files_to_project(files, project):
    for file in files:
        if file.name.endswith('.pdf'):
            file_instance = ProjectFile(project=project)
            file_instance.file_pdf = file  # PDF
            file = pdf_to_text(file)
            file_instance.file = file  # TXT
            file_instance.save()
        elif file.name.endswith('.txt'):
            file_instance = ProjectFile(file=file, project=project)
            file_instance.save()



def download(request, pk):
    if request.method == 'POST':

        file_ids = request.POST.getlist('files_to_download[]')
        files = ProjectFile.objects.filter(project_id=pk, id__in=file_ids)

        # for file_id in file_ids:
        #     print(files)
        #     print(file_id)
        #     files_to_zipped.append(files[int(file_id)])

        """Download archive zip file of code snippets"""
        response = HttpResponse(content_type='application/zip')
        zf = zipfile.ZipFile(response, 'w')

        # create the zipfile in memory using writestr
        # add a readme
        zf.writestr("README.MD", f'''# File List Total {len(files)} files. ''')
        project = get_object_or_404(Project, pk=pk)

        for file in files:
            if file.file_pdf != "":  # if pdf exists
                file_path = 'media/' + file.get_project_folder(file.filename_pdf())
                fdir, fname = os.path.split(file_path)
                zip_subdir = "pdfs"
                zip_path = os.path.join(zip_subdir, fname)
                # Add file, at correct path
                zf.write(file_path, zip_path)

            if file.file is not None:
                file_path = 'media/' + file.get_project_folder(file.filename())
                fdir, fname = os.path.split(file_path)
                zip_subdir = "txts"
                zip_path = os.path.join(zip_subdir, fname)
                # Add file, at correct path
                zf.write(file_path, zip_path)

        ZIPFILE_NAME = f"{project.project_folder}.zip"
        # return as zipfile
        response['Content-Disposition'] = f'attachment; filename={ZIPFILE_NAME}'
        return response


def full_view_to_file(request, file_pk):
    file = get_object_or_404(ProjectFile, pk=file_pk)
    return HttpResponse(file.preview(full_view=True))