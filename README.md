# ComfyUI 모델 관리자

ComfyUI 모델 관리자는 HuggingFace와 Civitai에서 모델을 쉽게 다운로드하고 관리할 수 있는 CLI(Command Line Interface) 도구입니다.

## 기능

- HuggingFace에서 모델 다운로드
- Civitai에서 모델 다운로드
- 사용자 정의 저장소 관리
- 모델 목록 보기 및 관리
- 직접 URL에서 다운로드
- 일괄 다운로드 기능
- 모델 무결성 검증
- 다운로드 진행률 및 속도 표시
- 다운로드 재시도 및 오류 복구 기능

## 설치 방법

1. 이 저장소를 클론하거나 다운로드합니다.
2. 필요한 패키지를 설치합니다:

```bash
pip install huggingface_hub requests tqdm
```

## 사용 방법

### 명령행 인터페이스 (CLI)

```bash
python comfyui_model_manager.py [command] [options]
```

### 대화형 모드

인자 없이 실행하면 대화형 메뉴가 표시됩니다:

```bash
python comfyui_model_manager.py
```

## 주요 명령어

### 설정 관리

```bash
# 설정 보기
python comfyui_model_manager.py config --show

# 설정 변경
python comfyui_model_manager.py config --set base_path /path/to/models
python comfyui_model_manager.py config --set civitai_token your_token
python comfyui_model_manager.py config --set concurrent_downloads 3
```

### HuggingFace 모델 관리

```bash
# 사용 가능한 모델 목록 보기
python comfyui_model_manager.py huggingface --list

# 모델 다운로드
python comfyui_model_manager.py huggingface --download MODEL_ID
```

### Civitai 모델 다운로드

```bash
# URL로 모델 다운로드
python comfyui_model_manager.py civitai --url https://civitai.com/models/12345
```

### 사용자 정의 저장소 관리

```bash
# 저장소 목록 보기
python comfyui_model_manager.py repo --list

# 저장소 추가
python comfyui_model_manager.py repo --add NAME REPO_ID "pattern1,pattern2" dir_type

# 저장소 삭제
python comfyui_model_manager.py repo --remove NAME

# 저장소에서 다운로드
python comfyui_model_manager.py repo --download NAME
```

### 직접 다운로드

```bash
# URL에서 직접 다운로드
python comfyui_model_manager.py download --url URL --dir dir_type [--filename FILENAME]
```

### 일괄 다운로드

```bash
# 여러 파일 일괄 다운로드
python comfyui_model_manager.py batch --file download_list.txt
```

`download_list.txt` 형식:

```
URL,dir_type[,filename]
https://example.com/model1.safetensors,checkpoints,my_model.safetensors
https://example.com/model2.safetensors,loras
```

### 모델 관리

```bash
# 설치된 모델 목록 보기
python comfyui_model_manager.py list

# 모델 무결성 검증
python comfyui_model_manager.py verify
```

## 모델 디렉토리 구조

모델은 기본 경로 아래의 해당 디렉토리에 저장됩니다:

- `checkpoints`: 체크포인트 모델
- `loras`: LoRA 모델
- `vae`: VAE 모델
- `controlnet`: ControlNet 모델
- 기타 다양한 모델 유형 지원

## 설정 옵션

- `base_path`: 모델 저장 기본 경로
- `civitai_token`: Civitai API 토큰
- `download_retries`: 다운로드 실패 시 재시도 횟수
- `download_timeout`: 다운로드 타임아웃 시간(초)
- `concurrent_downloads`: 동시 다운로드 수

## 에러 해결

1. HuggingFace 다운로드 오류:

   - `huggingface-cli login` 명령으로 로그인하여 인증합니다.
   - 올바른 저장소 ID를 확인합니다.

2. Civitai 다운로드 오류:

   - Civitai API 토큰이 올바르게 설정되었는지 확인합니다.
   - URL 형식이 올바른지 확인합니다.

3. 다운로드 중단 문제:
   - `verify` 명령을 사용하여 중단된 파일을 확인하고 복구합니다.

## 폴더 구조

모델은 다음 폴더 구조로 저장됩니다:

- `checkpoints/`: 메인 모델 체크포인트
- `loras/`: LoRA 모델
- `vae/`: VAE 모델
- `controlnet/`: ControlNet 모델
- `upscale_models/`: 업스케일 모델
- `clip/`: CLIP 모델
- `text_encoders/`: 텍스트 인코더
- `clip_vision/`: CLIP Vision 모델
- 기타 여러 폴더들...

## HuggingFace 모델 목록 관리

HuggingFace 모델 목록은 `huggingface_models.json` 파일에서 관리됩니다. 이 파일을 수정하여 새로운 모델을 추가하거나 기존 모델을 수정할 수 있습니다. 파일 구조는 다음과 같습니다:

```json
{
  "models": {
    "1": {
      "repo_id": "저장소/이름",
      "patterns": "파일패턴.safetensors",
      "dir_type": "저장_디렉토리",
      "description": "모델 설명"
    }
  }
}
```

- `repo_id`: HuggingFace 저장소 ID
- `patterns`: 다운로드할 파일 패턴 (문자열 또는 문자열 배열)
- `dir_type`: 모델이 저장될 디렉토리 유형
- `description`: 모델에 대한 설명

## Civitai API 토큰 설정

Civitai 모델을 다운로드하려면 API 토큰이 필요합니다:

1. [Civitai 계정](https://civitai.com/user/account)에 로그인
2. API 키 생성
3. 다음과 같이 설정:
   ```bash
   python comfyui_model_manager.py config --set civitai_token "your_token_here"
   ```

## 기여

이슈와 풀 리퀘스트를 환영합니다!

## 라이센스

MIT
