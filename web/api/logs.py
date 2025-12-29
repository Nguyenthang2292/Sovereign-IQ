"""
API routes for reading log files.
"""

from typing import Literal
from fastapi import APIRouter, HTTPException, Query, Request

from web.utils.log_manager import get_log_manager
from modules.common.ui.logging import log_error
from web.utils.translations import translate, get_locale_from_header
from web.utils.error_sanitizer import sanitize_error

router = APIRouter()


@router.get("/logs/{session_id}")
async def get_logs(
    session_id: str,
    request: Request,
    offset: int = Query(0, ge=0, description="Byte offset to start reading from"),
    command_type: Literal["scan", "analyze"] = Query(
        "scan",
        description="Type of command: 'scan' or 'analyze'"
    )
):
    """
    Get log content from a log file.
    
    Args:
        session_id: Unique session identifier
        request: Request - The incoming HTTP request (used for auth/context or routing)
        offset: Byte offset to start reading from
        command_type: Type of command ('scan' or 'analyze')
        
    Returns:
        Dict with logs content, new offset, and has_more flag
    """
    try:
        log_manager = get_log_manager()
        
        # Read log from offset
        log_content, new_offset = log_manager.read_log(
            session_id,
            offset,
            command_type
        )
        
        # Get file size to determine if there's more content
        file_size = log_manager.get_log_size(session_id, command_type)
        has_more = new_offset < file_size
        
        return {
            "success": True,
            "logs": log_content,
            "offset": new_offset,
            "has_more": has_more,
            "file_size": file_size
        }
    
    except Exception as e:
        log_error(f"Error reading logs for session {session_id}: {e}")
        
        # Get locale from request headers
        locale = get_locale_from_header(request.headers.get("Accept-Language"))
        
        # Sanitize error before displaying to user
        sanitized_error = sanitize_error(e)
        
        # Use translation function with sanitized error
        error_message = translate("errors.logReadError", locale=locale, error=sanitized_error)
        raise HTTPException(
            status_code=500,
            detail=error_message
        )
