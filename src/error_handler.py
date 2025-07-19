import logging
from typing import Optional, Dict, Any
from functools import wraps
import traceback

# Configure logging
logger = logging.getLogger(__name__)

class AppError(Exception):
    """Base exception class for application errors"""
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class DocumentProcessingError(AppError):
    """Exception for document processing errors"""
    def __init__(self, message: str, filename: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="DOCUMENT_PROCESSING_ERROR",
            details={"filename": filename}
        )

class EmbeddingError(AppError):
    """Exception for embedding generation errors"""
    def __init__(self, message: str, text_preview: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="EMBEDDING_ERROR",
            details={"text_preview": text_preview[:100] if text_preview else None}
        )

class QueryError(AppError):
    """Exception for query processing errors"""
    def __init__(self, message: str, query: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="QUERY_ERROR",
            details={"query": query}
        )

def handle_errors(default_response: Any = None, log_error: bool = True):
    """Decorator to handle errors gracefully"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AppError as e:
                if log_error:
                    logger.error(f"Application error in {func.__name__}: {e.message}", 
                               extra={"error_code": e.error_code, "details": e.details})
                if default_response is not None:
                    return default_response
                raise
            except Exception as e:
                if log_error:
                    logger.error(f"Unexpected error in {func.__name__}: {str(e)}", 
                               extra={"traceback": traceback.format_exc()})
                if default_response is not None:
                    return default_response
                raise AppError(f"Unexpected error in {func.__name__}: {str(e)}")
        return wrapper
    return decorator

def safe_execute(func, default_value=None, error_message="Operation failed"):
    """Safely execute a function with error handling"""
    try:
        return func()
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}")
        return default_value

def validate_file_type(filename: str, allowed_extensions: Optional[list] = None) -> bool:
    """Validate file type based on extension"""
    if allowed_extensions is None:
        allowed_extensions = ['.pdf', '.docx', '.txt']

    if not filename:
        return False

    # Get extension with dot, e.g. '.pdf'
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in [ext.lower() for ext in allowed_extensions]
import os  # Needed for os.path.splitext in validate_file_type

def validate_query(query: str, min_length: int = 3, max_length: int = 1000) -> bool:
    """Validate query parameters"""
    if not query or not isinstance(query, str):
        return False
    
    query = query.strip()
    return min_length <= len(query) <= max_length

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues"""
    if not filename:
        return "unknown_file"
    
    # Remove path separators and potentially dangerous characters
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = sanitized.strip('. ')  # Remove leading/trailing dots and spaces
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:250] + ('.' + ext if ext else '')
    
    return sanitized or "unknown_file"

def format_error_response(error: Exception, include_details: bool = False) -> Dict[str, Any]:
    """Format error for API response"""
    if isinstance(error, AppError):
        response: Dict[str, Any] = {
            "status": "error",
            "error_code": error.error_code,
            "message": error.message
        }
        if include_details and error.details:
            response["details"] = error.details  # type: ignore
    else:
        response: Dict[str, Any] = {
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred"
        }
        if include_details:
            response["details"] = {"error_type": type(error).__name__}  # type: ignore
    return response

def setup_error_logging():
    """Setup comprehensive error logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific log levels
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)