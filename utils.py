import json
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def validate_json_response(response: str) -> Tuple[bool, Optional[str]]:
    """
    모델의 JSON 응답을 검증하고 재시도 여부를 결정합니다.
    
    Args:
        response (str): 모델이 생성한 응답 텍스트
        
    Returns:
        Tuple[bool, Optional[str]]: (유효성 여부, 오류 메시지)
    """
    try:
        # JSON 파싱 전에 응답 정리
        response = response.strip()
        if not response.startswith('{') or not response.endswith('}'):
            return False, "응답이 올바른 JSON 형식이 아님"

        # 잘못된 형식 검사
        if "_json_" in response or "__metadata" in response or "@odata" in response:
            return False, "잘못된 JSON 형식 (메타데이터 포함)"

        data = json.loads(response)
        required_keys = ["question", "passage", "options", "answer", "explanation"]
        
        # 필수 키 확인
        if not all(key in data for key in required_keys):
            missing_keys = [key for key in required_keys if key not in data]
            return False, f"응답에 필수 키가 누락됨: {missing_keys}"

        # 추가 필드 확인
        extra_keys = [key for key in data.keys() if key not in required_keys]
        if extra_keys:
            return False, f"불필요한 필드가 포함됨: {extra_keys}"
            
        # 빈칸 개수 확인
        blanks = ["(A){{|bold-underline|}}", "(B){{|bold-underline|}}", "(C){{|bold-underline|}}"]
        blank_count = sum(1 for blank in blanks if blank in data["passage"])
        if blank_count != 3:  # 정확히 3개의 빈칸이 있어야 함
            return False, f"빈칸 개수가 정확하지 않음: {blank_count} (필요: 3)"
            
        # 옵션 형식 확인
        options = data["options"].split("//")
        if len(options) != 5:  # 정확히 5개의 옵션이 있어야 함
            return False, f"옵션 개수가 정확하지 않음: {len(options)} (필요: 5)"
            
        # 각 옵션의 형식 확인
        for i, option in enumerate(options):
            phrases = option.split(" - ")
            if len(phrases) != 3:  # 각 옵션은 정확히 3개의 구문이 있어야 함
                return False, f"옵션 {i+1}의 구문 개수가 정확하지 않음: {len(phrases)} (필요: 3)"
                
        return True, None
        
    except json.JSONDecodeError:
        return False, "JSON 파싱 실패"
    except Exception as e:
        return False, f"응답 검증 중 예외 발생: {str(e)}"

def extract_json_from_response(response_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    응답 텍스트에서 JSON 객체를 추출합니다.
    
    Args:
        response_text (str): 모델이 생성한 전체 응답 텍스트
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (추출된 JSON 문자열, 오류 메시지)
    """
    try:
        first_json_start = response_text.find('{')
        first_json_end = response_text.find('}', first_json_start) + 1
        if first_json_start != -1 and first_json_end != -1:
            return response_text[first_json_start:first_json_end], None
        return None, "JSON 객체를 찾을 수 없음"
    except Exception as e:
        return None, f"JSON 추출 중 오류: {str(e)}"

def should_retry_response(response_text: str, attempt: int, max_retries: int) -> Tuple[bool, Optional[str]]:
    """
    응답을 재시도할지 결정합니다.
    
    Args:
        response_text (str): 모델이 생성한 응답 텍스트
        attempt (int): 현재 시도 횟수
        max_retries (int): 최대 재시도 횟수
        
    Returns:
        Tuple[bool, Optional[str]]: (재시도 여부, 오류 메시지)
    """
    if not response_text:
        return True, f"빈 응답 (시도 {attempt + 1}/{max_retries})"
        
    # JSON 추출
    json_text, error = extract_json_from_response(response_text)
    if error:
        return True, error
        
    # JSON 검증
    is_valid, error = validate_json_response(json_text)
    if not is_valid:
        if attempt < max_retries - 1:
            return True, error
        return False, f"마지막 시도에서 유효하지 않은 응답: {error}"
        
    return False, None 