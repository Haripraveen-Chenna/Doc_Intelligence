from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional

from doc_tracker_service import (
    read_file,
    club_tasks,
    identify_doc_type,
    generate_doc_project
)

router = APIRouter()


@router.post("/generate-business-tasks")
async def generate_tasks(file: UploadFile = File(...)):
    try:
        file_format = await read_file(file)
        result = await club_tasks(file_format)
        return result  # already a clean dict: {"tasks": ..., "meta": ...}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-project")
async def generate_project(
    file: UploadFile = File(...),
    project_name: Optional[str] = None,
    project_description: Optional[str] = None,
    user_data: Optional[str] = None
):
    user_data = user_data or "I have a bakery that I run with my wife in Philadelphia. I do not have any current investment."
    project_name = project_name or "Marketing"
    project_description = project_description or "I want a project to market this event properly using the sheet that I have shared."

    try:
        file_format = await read_file(file)
        doc_type = await identify_doc_type(file_format)

        result = await generate_doc_project(
            format=file_format,
            project_name=project_name,
            project_description=project_description,
            user_data=user_data,
            doc_type=doc_type
        )

        return result  # already a clean dict: {"projects": ..., "meta": ...}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))