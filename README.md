# ComfyUI 모델 관리자

ComfyUI 모델 관리자는 RunPod 및 기타 환경에서 ComfyUI에 필요한 모델을 쉽게 다운로드하고 관리할 수 있는 CLI 도구입니다.

## 특징

- HuggingFace와 Civitai에서 모델 다운로드
- 사전 정의된 인기 모델 목록 (외부 JSON 파일에서 관리)
- 사용자 정의 저장소 추가 기능
- 설치된 모델 목록 보기
- 모델 경로 및 기타 설정 관리
- CLI와 인터랙티브 모드 모두 지원

## 설치

```bash
# 저장소 클론
git clone https://github.com/NewTurn2017/comfyui-model-manager.git
cd comfyui-model-manager

# 필요한 의존성 설치
pip install huggingface_hub>=0.25.2 hf_transfer>=0.1.8 requests>=2.25.0
```

## 사용법

### CLI 모드

```bash
# 도움말 보기
python comfyui_model_manager.py --help

# HuggingFace 모델 목록 보기
python comfyui_model_manager.py huggingface --list

# HuggingFace 모델 다운로드
python comfyui_model_manager.py huggingface --download 1

# Civitai 모델 다운로드
python comfyui_model_manager.py civitai --url "https://civitai.com/models/12345"

# 사용자 정의 저장소 추가
python comfyui_model_manager.py repo --add "내저장소" "username/repo" "*.safetensors" "checkpoints"

# 사용자 정의 저장소에서 다운로드
python comfyui_model_manager.py repo --download "내저장소"

# 설치된 모델 목록 보기
python comfyui_model_manager.py list

# 설정 보기
python comfyui_model_manager.py config --show

# 설정 업데이트
python comfyui_model_manager.py config --set base_path "/새로운/경로"
```

### 인터랙티브 모드

```bash
python comfyui_model_manager.py
```

이것은 대화형 메뉴를 표시하여 다양한 작업을 수행할 수 있습니다.

## 설정

설정은 `~/.comfyui_model_manager.json`에 저장됩니다. 다음 설정을 구성할 수 있습니다:

- `base_path`: 모델이 저장되는 기본 경로 (기본값: `/workspace/ComfyUI/models`)
- `civitai_token`: Civitai API 토큰 (Civitai 모델을 다운로드하려면 필요)
- `custom_repos`: 사용자 정의 저장소 목록

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
