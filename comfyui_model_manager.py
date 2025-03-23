#!/usr/bin/env python3
# comfyui_model_manager.py
# ComfyUI 모델 관리자 - HuggingFace 및 Civitai 모델 다운로드 도구

import os
import sys
import re
import json
import argparse
import subprocess
import requests
import time
from pathlib import Path
import shutil
from typing import List, Dict, Union, Optional, Any, Tuple
import concurrent.futures
from tqdm import tqdm

# 기본 설정
DEFAULT_CONFIG = {
    "base_path": "/home/elicer/ComfyUI/models",  # 기본 모델 경로
    "civitai_token": "",  # Civitai API 토큰
    "custom_repos": {},  # 사용자 정의 저장소
    "download_retries": 3,  # 다운로드 재시도 횟수
    "download_timeout": 120,  # 다운로드 타임아웃 시간(초)
    "concurrent_downloads": 2  # 동시 다운로드 수
}

# 모델 디렉토리 구조
MODEL_DIRS = {
    "vae": "vae",
    "unet": "unet",
    "checkpoints": "checkpoints",
    "clip": "clip",
    "text_encoders": "text_encoders",
    "controlnet": "controlnet",
    "style_models": "style_models",
    "loras": "loras",
    "upscale_models": "upscale_models",
    "clip_vision": "clip_vision",
    "pulid": "pulid",
    "embeddings": "embeddings",
    "hypernetworks": "hypernetworks",
    "animatediff_models": "animatediff_models",
    "inpaint": "inpaint",
    "animatediff_motion_lora": "animatediff_motion_lora",
    "annotator": "annotator",
    "configs": "configs",
    "detection": "detection",
    "diffusers": "diffusers",
    "diffusion_models": "diffusion_models",
    "facerestore_models": "facerestore_models",
    "gligen": "gligen",
    "insightface": "insightface",
    "inisghtface": "inisghtface",
    "ipadapter": "ipadapter",
    "nsfw_detector": "nsfw_detector",
    "onnx": "onnx",
    "photomaker": "photomaker",
    "reactor": "reactor",
    "sams": "sams",
    "vae_approx": "vae_approx"
}

# Civitai 모델 타입 매핑
CIVITAI_TYPE_MAP = {
    'checkpoint': 'checkpoints',
    'lora': 'loras',
    'textualinversion': 'embeddings',
    'hypernetwork': 'hypernetworks',
    'controlnet': 'controlnet',
    'vae': 'vae',
    'upscaler': 'upscale_models'
}


class DownloadManager:
    """파일 다운로드를 관리하는 클래스"""

    def __init__(self, config: Dict[str, Any]):
        self.retries = config.get("download_retries", 3)
        self.timeout = config.get("download_timeout", 120)
        self.concurrent_downloads = config.get("concurrent_downloads", 2)

    def download_file(self, url: str, path: Path, filename: Optional[str] = None) -> Optional[Path]:
        """
        파일을 다운로드하는 함수

        Args:
            url: 다운로드할 파일의 URL
            path: 파일을 저장할 경로
            filename: 저장할 파일명 (None이면 URL에서 추출)

        Returns:
            다운로드된 파일 경로 또는 실패 시 None
        """
        # 디렉토리가 없으면 생성
        path.mkdir(parents=True, exist_ok=True)

        # 파일명이 제공되지 않은 경우 URL에서 추출
        if not filename:
            response = requests.head(url, allow_redirects=True)
            content_disposition = response.headers.get('content-disposition')
            if content_disposition:
                filename_match = re.search(
                    r'filename="?([^"]+)"?', content_disposition)
                if filename_match:
                    filename = filename_match.group(1)

            if not filename:
                filename = os.path.basename(url)

        file_path = path / filename
        temp_path = file_path.with_suffix(file_path.suffix + '.part')

        # 이미 완료된 파일이 있는지 확인
        if file_path.exists():
            print(f"파일이 이미 존재합니다: {file_path}")
            return file_path

        downloaded_size = 0
        if temp_path.exists():
            downloaded_size = temp_path.stat().st_size

        headers = {}
        if downloaded_size:
            headers['Range'] = f"bytes={downloaded_size}-"

        for attempt in range(self.retries + 1):
            try:
                response = requests.get(
                    url, headers=headers, stream=True, timeout=self.timeout)
                response.raise_for_status()

                # 파일 크기 계산
                total_size = downloaded_size
                if 'content-length' in response.headers:
                    total_size += int(response.headers.get('content-length', 0))

                # 이어받기 모드로 파일 열기
                write_mode = 'ab' if downloaded_size else 'wb'

                with open(temp_path, write_mode) as file:
                    with tqdm(
                        desc=f"다운로드 중: {filename}",
                        initial=downloaded_size,
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress_bar:
                        start_time = time.time()
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                                progress_bar.update(len(chunk))

                                # 속도 계산 및 표시
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    speed = progress_bar.n / elapsed / 1024  # KB/s
                                    progress_bar.set_postfix(
                                        {"속도": f"{speed:.2f} KB/s"})

                # 다운로드 완료 후 파일 이름 변경
                temp_path.rename(file_path)
                print(f"다운로드 완료: {file_path}")
                return file_path

            except (requests.RequestException, IOError) as e:
                if attempt < self.retries:
                    wait_time = 2 ** attempt  # 지수 백오프
                    print(
                        f"다운로드 오류: {e}. {wait_time}초 후 재시도... ({attempt+1}/{self.retries})")
                    time.sleep(wait_time)
                else:
                    print(f"다운로드 실패: {e}. 최대 재시도 횟수 초과.")
                    return None

        return None

    def download_files(self, urls: List[Tuple[str, Path, Optional[str]]], max_workers: Optional[int] = None) -> List[Optional[Path]]:
        """
        여러 파일을 병렬로 다운로드

        Args:
            urls: (url, path, filename) 튜플의 리스트
            max_workers: 최대 동시 다운로드 수

        Returns:
            다운로드된 파일 경로 리스트
        """
        if max_workers is None:
            max_workers = self.concurrent_downloads

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.download_file, url, path, filename): (url, path, filename)
                       for url, path, filename in urls}

            for future in concurrent.futures.as_completed(futures):
                url, path, filename = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"다운로드 중 예외 발생: {e}")
                    results.append(None)

        return results


class ComfyUIModelManager:
    def __init__(self):
        self.config = self._load_config()
        self.base_path = Path(self.config["base_path"])
        self.huggingface_models = {}
        self.huggingface_descriptions = {}
        self._load_huggingface_models()
        self._ensure_dependencies()
        self._ensure_directories_exist()

        # 다운로드 관리자 초기화
        self.download_manager = DownloadManager(self.config)

    def _load_config(self) -> Dict[str, Any]:
        """설정 파일을 로드하거나 기본 설정을 사용합니다."""
        config_path = Path(__file__).parent / "huggingface_models.json"

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # models 키가 있는지 확인 (huggingface_models.json 파일 구조)
                    if "models" in data:
                        # 기존 models 정보 보존
                        models_data = data.get("models", {})

                        # config 정보가 있으면 로드, 없으면 기본값
                        config = data.get("config", DEFAULT_CONFIG)

                        # 기본 설정에 없는 키가 있으면 추가
                        for key, value in DEFAULT_CONFIG.items():
                            if key not in config:
                                config[key] = value

                        return config
                    else:
                        # 기존 형식이 아닌 경우 그대로 사용
                        config = data
                        # 기본 설정에 없는 키가 있으면 추가
                        for key, value in DEFAULT_CONFIG.items():
                            if key not in config:
                                config[key] = value
                        return config
            except Exception as e:
                print(f"설정 파일 로드 오류: {e}")
                print("기본 설정을 사용합니다.")

        # 설정 파일이 없으면 기본 설정 생성
        try:
            # 이미 huggingface_models.json 파일이 없는 경우 새로 생성
            with open(config_path, 'w', encoding='utf-8') as f:
                # 초기 구조는 models 정보와 config 정보를 포함
                data = {
                    "models": {},
                    "config": DEFAULT_CONFIG
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"설정 파일 생성 오류: {e}")

        return DEFAULT_CONFIG

    def _load_huggingface_models(self):
        """HuggingFace 모델 데이터를 JSON 파일에서 로드합니다."""
        # 현재 스크립트 위치를 기준으로 상대 경로 사용
        models_path = Path(__file__).parent / "huggingface_models.json"

        if not models_path.exists():
            print("경고: huggingface_models.json 파일을 찾을 수 없습니다.")
            return

        try:
            with open(models_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if "models" in data:
                    for model_id, model_info in data["models"].items():
                        model_id = int(model_id)
                        self.huggingface_models[model_id] = (
                            model_info["repo_id"],
                            model_info["patterns"],
                            model_info["dir_type"]
                        )
                        self.huggingface_descriptions[model_id] = model_info["description"]
        except Exception as e:
            print(f"HuggingFace 모델 데이터 로드 오류: {e}")

    def _save_config(self):
        """현재 설정을 파일에 저장합니다."""
        config_path = Path(__file__).parent / "huggingface_models.json"

        try:
            # 기존 파일이 있으면 내용 읽기
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"models": {}}

            # config 업데이트
            data["config"] = self.config

            # 파일에 저장
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("설정이 저장되었습니다.")
        except Exception as e:
            print(f"설정 저장 오류: {e}")

    def _ensure_dependencies(self):
        """필요한 의존성을 설치합니다."""
        required_packages = ["huggingface_hub>=0.25.2",
                             "hf_transfer>=0.1.8",
                             "requests>=2.25.0",
                             "tqdm>=4.66.0"]

        for package in required_packages:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"패키지 설치 오류 ({package}): {e}")

    def _ensure_directories_exist(self):
        """필요한 모든 디렉토리가 존재하는지 확인합니다."""
        for directory in MODEL_DIRS.values():
            path = self.base_path / directory
            path.mkdir(parents=True, exist_ok=True)

    def download_huggingface_model(self, model_id: int):
        """HuggingFace에서 모델을 다운로드합니다."""
        try:
            from huggingface_hub import snapshot_download, model_info
        except ImportError:
            print("huggingface_hub 패키지가 설치되어 있지 않습니다.")
            self._ensure_dependencies()
            from huggingface_hub import snapshot_download, model_info

        if not self.huggingface_models:
            print("오류: HuggingFace 모델 데이터가 로드되지 않았습니다.")
            return False

        if model_id not in self.huggingface_models:
            print(f"오류: 모델 ID {model_id}가 존재하지 않습니다.")
            print("사용 가능한 모델 목록을 보려면 'huggingface --list' 명령을 사용하세요.")
            return False

        repo_id, patterns, dir_type = self.huggingface_models[model_id]
        target_dir = self.base_path / MODEL_DIRS[dir_type]

        description = self.huggingface_descriptions.get(model_id, "설명 없음")
        print(f"다운로드 중: {description} (repo: {repo_id})")

        try:
            # 저장소 정보 확인
            try:
                repo_info = model_info(repo_id)
                print(
                    f"모델 정보: {repo_info.modelId} (마지막 수정: {repo_info.lastModified})")
            except Exception as e:
                print(f"모델 정보 조회 실패: {e}")

            # 스냅샷 다운로드 사용
            if patterns:
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=patterns,
                    local_dir=str(target_dir),
                    tqdm_class=tqdm
                )
            else:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target_dir),
                    tqdm_class=tqdm
                )
            print(f"다운로드 완료: {target_dir}")
            return True
        except Exception as e:
            print(f"다운로드 오류: {e}")

            # 직접 다운로드 시도
            try:
                print("스냅샷 다운로드 실패, 직접 다운로드 시도 중...")
                return self._direct_download_from_huggingface(repo_id, patterns, target_dir)
            except Exception as direct_e:
                print(f"직접 다운로드도 실패: {direct_e}")
                return False

    def _direct_download_from_huggingface(self, repo_id: str, patterns: List[str], target_dir: Path) -> bool:
        """HuggingFace API를 통해 직접 파일을 다운로드합니다."""
        from huggingface_hub import list_repo_files, hf_hub_download

        try:
            # 저장소의 모든 파일 목록 가져오기
            all_files = list_repo_files(repo_id)
            files_to_download = []

            # 패턴이 있으면 필터링
            if patterns:
                for pattern in patterns:
                    import fnmatch
                    files_to_download.extend(
                        [f for f in all_files if fnmatch.fnmatch(f, pattern)])
            else:
                files_to_download = all_files

            if not files_to_download:
                print(f"다운로드할 파일을 찾을 수 없습니다: {repo_id}")
                return False

            print(f"다운로드할 파일: {len(files_to_download)}개")

            # 파일 다운로드
            for file_path in tqdm(files_to_download, desc=f"다운로드 중: {repo_id}"):
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        local_dir=str(target_dir)
                    )
                except Exception as e:
                    print(f"파일 다운로드 실패 ({file_path}): {e}")

            print(f"다운로드 완료: {target_dir}")
            return True
        except Exception as e:
            print(f"직접 다운로드 오류: {e}")
            return False

    def download_civitai_model(self, url: str):
        """Civitai에서 모델을 다운로드합니다."""
        if not self.config.get("civitai_token"):
            print("Civitai API 토큰이 설정되지 않았습니다.")
            token = input(
                "Civitai API 토큰을 입력하세요 (https://civitai.com/user/account 에서 생성): ")
            self.config["civitai_token"] = token
            self._save_config()

        model_id, version_id = self._extract_ids_from_url(url)

        if not version_id and not model_id:
            print("잘못된 URL입니다. modelVersionId 또는 modelId를 포함해야 합니다.")
            return False

        # modelId만 있는 경우 최신 버전 ID 가져오기
        if not version_id and model_id:
            version_id = self._get_latest_version_id(model_id)
            if not version_id:
                print("최신 버전 ID를 가져오지 못했습니다.")
                return False

        info = self._get_model_info(version_id)
        if not info:
            print("모델 정보를 가져오지 못했습니다.")
            return False

        model_type = info['type']
        dir_type = CIVITAI_TYPE_MAP.get(model_type, 'checkpoints')
        model_path = self.base_path / MODEL_DIRS[dir_type]

        # 다운로드 URL 생성
        download_url = f"https://civitai.com/api/download/models/{version_id}"

        # 다운로드 수행
        result = self.download_manager.download_file(
            url=download_url,
            path=model_path,
            filename=info['name']
        )

        if result:
            print(f"모델 다운로드 완료: {info['model_name']} - {info['name']}")
            print(f"저장 위치: {result}")
            return True
        else:
            print(f"모델 다운로드 실패: {info['model_name']}")
            return False

    def _extract_ids_from_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """URL에서 modelVersionId와 modelId를 추출합니다."""
        version_id = None
        model_id = None

        # modelVersionId 확인
        version_match = re.search(r'modelVersionId=(\d+)', url)
        if version_match:
            version_id = version_match.group(1)

        # modelId 확인
        model_match = re.search(r'models/(\d+)', url)
        if model_match:
            model_id = model_match.group(1)

        return model_id, version_id

    def _get_latest_version_id(self, model_id: str) -> Optional[str]:
        """모델 ID로 최신 버전 ID를 가져옵니다."""
        try:
            response = requests.get(
                f"https://civitai.com/api/v1/models/{model_id}",
                headers={
                    'Authorization': f'Bearer {self.config["civitai_token"]}'}
            )
            response.raise_for_status()
            data = response.json()
            # 최신 버전의 ID 반환
            return str(data['modelVersions'][0]['id'])
        except Exception as e:
            print(f"최신 버전 ID 가져오기 오류: {e}")
            return None

    def _get_model_info(self, version_id: str) -> Optional[Dict[str, str]]:
        """버전 ID로 모델 정보를 가져옵니다."""
        try:
            response = requests.get(
                f"https://civitai.com/api/v1/model-versions/{version_id}",
                headers={
                    'Authorization': f'Bearer {self.config["civitai_token"]}'}
            )
            response.raise_for_status()
            data = response.json()
            return {
                'name': data['files'][0]['name'],
                'type': data['model']['type'].lower(),
                'model_name': data['model']['name']
            }
        except Exception as e:
            print(f"모델 정보 가져오기 오류: {e}")
            return None

    def add_custom_repo(self, name: str, repo_id: str, patterns: Union[str, List[str]], dir_type: str):
        """사용자 정의 저장소를 추가합니다."""
        if dir_type not in MODEL_DIRS:
            print(f"오류: 잘못된 디렉토리 유형 '{dir_type}'")
            print(f"가능한 유형: {', '.join(MODEL_DIRS.keys())}")
            return False

        if isinstance(patterns, str):
            patterns = [patterns]

        self.config["custom_repos"][name] = {
            "repo_id": repo_id,
            "patterns": patterns,
            "dir_type": dir_type
        }
        self._save_config()
        print(f"사용자 정의 저장소가 추가되었습니다: {name}")
        return True

    def remove_custom_repo(self, name: str):
        """사용자 정의 저장소를 삭제합니다."""
        if name in self.config["custom_repos"]:
            del self.config["custom_repos"][name]
            self._save_config()
            print(f"사용자 정의 저장소가 삭제되었습니다: {name}")
            return True
        else:
            print(f"오류: 저장소 '{name}'를 찾을 수 없습니다.")
            return False

    def download_custom_repo(self, name: str):
        """사용자 정의 저장소에서 모델을 다운로드합니다."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub 패키지가 설치되어 있지 않습니다.")
            self._ensure_dependencies()
            from huggingface_hub import snapshot_download

        # 다운로드할 저장소 목록을 해석
        repo_names = []

        # 콤마로 구분된 형식 (예: "1,3,5")
        if "," in name:
            parts = [part.strip() for part in name.split(",")]
            for part in parts:
                try:
                    # 숫자인 경우 인덱스로 처리
                    idx = int(part) - 1
                    repos = list(self.config["custom_repos"].keys())
                    if 0 <= idx < len(repos):
                        repo_names.append(repos[idx])
                    else:
                        print(f"오류: 인덱스 '{part}'가 범위를 벗어납니다.")
                except ValueError:
                    # 이름인 경우 그대로 사용
                    if part in self.config["custom_repos"]:
                        repo_names.append(part)
                    else:
                        print(f"오류: 저장소 '{part}'를 찾을 수 없습니다.")

        # 범위 형식 (예: "1-3")
        elif "-" in name:
            try:
                start, end = map(int, name.split("-"))
                repos = list(self.config["custom_repos"].keys())
                for idx in range(start-1, end):
                    if 0 <= idx < len(repos):
                        repo_names.append(repos[idx])
                    else:
                        print(f"오류: 인덱스 '{idx+1}'가 범위를 벗어납니다.")
            except ValueError:
                print(f"오류: 범위 '{name}'의 형식이 잘못되었습니다. 예: '1-3'")

        # 단일 저장소
        else:
            try:
                # 숫자인 경우 인덱스로 처리
                idx = int(name) - 1
                repos = list(self.config["custom_repos"].keys())
                if 0 <= idx < len(repos):
                    repo_names.append(repos[idx])
                else:
                    print(f"오류: 인덱스 '{name}'가 범위를 벗어납니다.")
            except ValueError:
                # 이름인 경우 그대로 사용
                if name in self.config["custom_repos"]:
                    repo_names.append(name)
                else:
                    print(f"오류: 저장소 '{name}'를 찾을 수 없습니다.")

        if not repo_names:
            print("다운로드할 저장소가 없습니다.")
            return False

        # 선택된 저장소 목록 표시
        print(f"\n다운로드할 저장소: {len(repo_names)}개")
        for repo_name in repo_names:
            repo = self.config["custom_repos"][repo_name]
            print(f"- {repo_name} (Repo ID: {repo['repo_id']})")

        # 병렬 다운로드 함수 정의
        def download_repo(repo_name):
            if repo_name not in self.config["custom_repos"]:
                print(f"오류: 저장소 '{repo_name}'를 찾을 수 없습니다.")
                return False

            repo = self.config["custom_repos"][repo_name]
            repo_id = repo["repo_id"]
            patterns = repo["patterns"]
            dir_type = repo["dir_type"]
            target_dir = self.base_path / MODEL_DIRS[dir_type]

            print(f"다운로드 시작: {repo_name} (repo: {repo_id})")

            try:
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=patterns,
                    local_dir=str(target_dir),
                    tqdm_class=tqdm
                )
                print(f"다운로드 완료: {repo_name} → {target_dir}")
                return True
            except Exception as e:
                print(f"스냅샷 다운로드 오류 ({repo_name}): {e}")
                try:
                    print(f"직접 다운로드 시도 중...")
                    return self._direct_download_from_huggingface(repo_id, patterns, target_dir)
                except Exception as direct_e:
                    print(f"직접 다운로드 실패: {direct_e}")
                    return False

        # 병렬로 다운로드 실행
        max_workers = min(self.config.get(
            "concurrent_downloads", 2), len(repo_names))
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_repo = {executor.submit(
                download_repo, repo_name): repo_name for repo_name in repo_names}
            for future in concurrent.futures.as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"다운로드 실패 ({repo_name}): {e}")
                    results.append(False)

        # 결과 요약
        success_count = results.count(True)
        print(f"\n다운로드 결과: {success_count}/{len(repo_names)} 성공")

        return success_count > 0

    def list_models(self):
        """모든 모델 디렉토리의 모델을 나열합니다."""
        model_counts = {}
        total_size = 0

        for dir_type, dir_name in MODEL_DIRS.items():
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                files = list(dir_path.glob('*.*'))
                if files:
                    model_counts[dir_type] = {
                        'count': len(files),
                        # MB
                        'size': sum(file.stat().st_size for file in files) / (1024 * 1024)
                    }
                    total_size += model_counts[dir_type]['size']

        # 결과 출력
        print("\n=== 설치된 모델 목록 ===")
        for dir_type, info in sorted(model_counts.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"[{dir_type}] - {info['count']}개 파일 ({info['size']:.2f} MB)")

        print(
            f"\n총계: {sum(info['count'] for info in model_counts.values())}개 파일 ({total_size:.2f} MB)")

        # 세부 목록 표시 여부 확인
        show_details = input("\n세부 목록을 보시겠습니까? (y/n): ").lower() == 'y'
        if show_details:
            for dir_type, dir_name in MODEL_DIRS.items():
                dir_path = self.base_path / dir_name
                if dir_path.exists():
                    files = list(dir_path.glob('*.*'))
                    if files:
                        print(f"\n[{dir_type}] - {len(files)}개 파일:")
                        for file in files:
                            size_mb = file.stat().st_size / (1024 * 1024)
                            print(f"  - {file.name} ({size_mb:.2f} MB)")

    def verify_models(self):
        """모델 파일의 무결성을 검증합니다."""
        print("\n모델 무결성 검증 중...")

        incomplete_files = []
        for dir_type, dir_name in MODEL_DIRS.items():
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                part_files = list(dir_path.glob('*.part'))
                if part_files:
                    for file in part_files:
                        incomplete_files.append(file)
                        print(f"미완료 다운로드 파일 발견: {file}")

        if incomplete_files:
            action = input(
                "\n미완료 파일을 어떻게 처리하시겠습니까? (삭제: d, 완료: c, 무시: i): ").lower()

            if action == 'd':
                for file in incomplete_files:
                    try:
                        file.unlink()
                        print(f"삭제됨: {file}")
                    except Exception as e:
                        print(f"삭제 실패: {file} - {e}")

            elif action == 'c':
                for file in incomplete_files:
                    try:
                        # 파일명에서 .part 제거
                        complete_path = file.with_suffix('')
                        file.rename(complete_path)
                        print(f"완료 처리됨: {file} → {complete_path}")
                    except Exception as e:
                        print(f"완료 처리 실패: {file} - {e}")
        else:
            print("모든 파일이 완전한 상태입니다.")

        return len(incomplete_files) == 0

    def update_config(self, key: str, value: str):
        """설정을 업데이트합니다."""
        if key == "base_path":
            # 경로가 유효한지 확인
            path = Path(value)
            if not path.exists():
                try:
                    path.mkdir(parents=True)
                except Exception as e:
                    print(f"디렉토리 생성 오류: {e}")
                    return False

            # 기존 경로에서 새 경로로 모델 이동 여부 확인
            if Path(self.config["base_path"]).exists():
                move = input(f"기존 모델을 새 경로({value})로 이동하시겠습니까? (y/n): ")
                if move.lower() == 'y':
                    try:
                        for dir_name in MODEL_DIRS.values():
                            src = Path(self.config["base_path"]) / dir_name
                            dst = path / dir_name
                            if src.exists():
                                if not dst.exists():
                                    dst.mkdir(parents=True, exist_ok=True)
                                for file in src.glob('*.*'):
                                    shutil.move(
                                        str(file), str(dst / file.name))
                        print("모델이 새 경로로 이동되었습니다.")
                    except Exception as e:
                        print(f"모델 이동 오류: {e}")
                        return False
        elif key == "concurrent_downloads":
            try:
                value = int(value)
                if value < 1:
                    print("동시 다운로드 수는 1 이상이어야 합니다.")
                    return False
            except ValueError:
                print("동시 다운로드 수는 정수여야 합니다.")
                return False
        elif key == "download_retries":
            try:
                value = int(value)
                if value < 0:
                    print("재시도 횟수는 0 이상이어야 합니다.")
                    return False
            except ValueError:
                print("재시도 횟수는 정수여야 합니다.")
                return False
        elif key == "download_timeout":
            try:
                value = int(value)
                if value < 1:
                    print("타임아웃 시간은 1 이상이어야 합니다.")
                    return False
            except ValueError:
                print("타임아웃 시간은 정수여야 합니다.")
                return False

        # 설정 업데이트
        self.config[key] = value
        self._save_config()

        # base_path 변경 시 인스턴스 변수도 업데이트
        if key == "base_path":
            self.base_path = Path(value)
            self._ensure_directories_exist()

        print(f"설정이 업데이트되었습니다: {key} = {value}")
        return True

    def direct_download(self, url: str, dir_type: str, filename: Optional[str] = None):
        """URL에서 직접 모델을 다운로드합니다."""
        if dir_type not in MODEL_DIRS:
            print(f"오류: 잘못된 디렉토리 유형 '{dir_type}'")
            print(f"가능한 유형: {', '.join(MODEL_DIRS.keys())}")
            return False

        target_dir = self.base_path / MODEL_DIRS[dir_type]

        result = self.download_manager.download_file(
            url=url,
            path=target_dir,
            filename=filename
        )

        if result:
            print(f"다운로드 완료: {result}")
            return True
        else:
            print("다운로드 실패")
            return False

    def batch_download(self, urls_file: str):
        """일괄 다운로드 기능을 제공합니다."""
        try:
            with open(urls_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            downloads = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(',')
                if len(parts) < 2:
                    print(f"잘못된 형식: {line}")
                    continue

                url = parts[0].strip()
                dir_type = parts[1].strip()

                filename = None
                if len(parts) >= 3:
                    filename = parts[2].strip()

                if dir_type not in MODEL_DIRS:
                    print(f"잘못된 디렉토리 유형: {dir_type} (행: {line})")
                    continue

                target_dir = self.base_path / MODEL_DIRS[dir_type]
                downloads.append((url, target_dir, filename))

            if not downloads:
                print("다운로드할 항목이 없습니다.")
                return False

            print(f"일괄 다운로드 시작: {len(downloads)}개 항목")
            results = self.download_manager.download_files(
                downloads,
                max_workers=self.config.get("concurrent_downloads", 2)
            )

            success_count = sum(1 for r in results if r is not None)
            print(f"\n다운로드 결과: {success_count}/{len(downloads)} 성공")

            return success_count > 0

        except Exception as e:
            print(f"일괄 다운로드 오류: {e}")
            return False


def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="ComfyUI 모델 관리자")

    subparsers = parser.add_subparsers(dest="command", help="사용할 명령")

    # 설정 명령
    config_parser = subparsers.add_parser("config", help="설정 관리")
    config_parser.add_argument(
        "--set", nargs=2, metavar=("KEY", "VALUE"), help="설정 업데이트")
    config_parser.add_argument("--show", action="store_true", help="현재 설정 표시")

    # HuggingFace 모델 다운로드 명령
    hf_parser = subparsers.add_parser(
        "huggingface", help="HuggingFace 모델 다운로드")
    hf_parser.add_argument("--list", action="store_true",
                           help="사용 가능한 HuggingFace 모델 나열")
    hf_parser.add_argument("--download", type=int,
                           metavar="MODEL_ID", help="모델 ID로 다운로드")

    # Civitai 모델 다운로드 명령
    civitai_parser = subparsers.add_parser("civitai", help="Civitai 모델 다운로드")
    civitai_parser.add_argument("--url", type=str, help="Civitai 모델 URL")

    # 사용자 정의 저장소 관리 명령
    repo_parser = subparsers.add_parser("repo", help="사용자 정의 저장소 관리")
    repo_parser.add_argument("--add", nargs=4, metavar=("NAME", "REPO_ID", "PATTERNS", "DIR_TYPE"),
                             help="사용자 정의 저장소 추가")
    repo_parser.add_argument("--remove", type=str,
                             metavar="NAME", help="사용자 정의 저장소 제거")
    repo_parser.add_argument(
        "--list", action="store_true", help="사용자 정의 저장소 나열")
    repo_parser.add_argument("--download", type=str,
                             metavar="NAME", help="사용자 정의 저장소에서 다운로드")

    # 모델 목록 명령
    list_parser = subparsers.add_parser("list", help="설치된 모델 나열")

    # 직접 다운로드 명령
    direct_parser = subparsers.add_parser("download", help="URL에서 직접 다운로드")
    direct_parser.add_argument(
        "--url", type=str, required=True, help="다운로드할 URL")
    direct_parser.add_argument(
        "--dir", type=str, required=True, help="저장할 디렉토리 유형")
    direct_parser.add_argument("--filename", type=str, help="저장할 파일명")

    # 일괄 다운로드 명령
    batch_parser = subparsers.add_parser("batch", help="일괄 다운로드")
    batch_parser.add_argument(
        "--file", type=str, required=True, help="다운로드 목록 파일")

    # 모델 검증 명령
    verify_parser = subparsers.add_parser("verify", help="모델 무결성 검증")

    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    manager = ComfyUIModelManager()

    if args.command == "config":
        if args.set:
            key, value = args.set
            manager.update_config(key, value)
        elif args.show:
            print("현재 설정:")
            for key, value in manager.config.items():
                if key == "custom_repos":
                    print(f"사용자 정의 저장소 ({len(value)}개):")
                    for name, repo in value.items():
                        print(
                            f"  - {name}: {repo['repo_id']} ({repo['dir_type']})")
                else:
                    print(f"  {key}: {value}")
        else:
            print("사용 가능한 설정 옵션:")
            for key, value in DEFAULT_CONFIG.items():
                print(f"  {key}: {value} (기본값)")

    elif args.command == "huggingface":
        if args.list:
            print("사용 가능한 HuggingFace 모델:")
            for id, desc in manager.huggingface_descriptions.items():
                repo_id = manager.huggingface_models[id][0]
                print(f"{id}: {desc} (repo: {repo_id})")
        elif args.download:
            manager.download_huggingface_model(args.download)
        else:
            print("huggingface --list 또는 huggingface --download MODEL_ID 명령을 사용하세요.")

    elif args.command == "civitai":
        if args.url:
            manager.download_civitai_model(args.url)
        else:
            print("Civitai URL을 제공해야 합니다.")

    elif args.command == "repo":
        if args.add:
            name, repo_id, patterns, dir_type = args.add
            patterns = patterns.split(',')
            manager.add_custom_repo(name, repo_id, patterns, dir_type)
        elif args.remove:
            manager.remove_custom_repo(args.remove)
        elif args.list:
            print("사용자 정의 저장소:")
            for name, repo in manager.config["custom_repos"].items():
                print(f"- {name}:")
                print(f"  Repo ID: {repo['repo_id']}")
                print(f"  패턴: {repo['patterns']}")
                print(f"  디렉토리: {repo['dir_type']}")
        elif args.download:
            manager.download_custom_repo(args.download)
        else:
            print("repo --list, repo --add, repo --remove 또는 repo --download 명령을 사용하세요.")

    elif args.command == "list":
        manager.list_models()

    elif args.command == "download":
        if args.url and args.dir:
            manager.direct_download(args.url, args.dir, args.filename)
        else:
            print("URL과 디렉토리 유형을 제공해야 합니다.")

    elif args.command == "batch":
        if args.file:
            manager.batch_download(args.file)
        else:
            print("다운로드 목록 파일을 제공해야 합니다.")

    elif args.command == "verify":
        manager.verify_models()

    else:
        # 인터랙티브 모드
        show_menu(manager)


def show_menu(manager):
    """인터랙티브 메뉴를 표시합니다."""
    while True:
        print("\n=== ComfyUI 모델 관리자 ===")
        print("1: HuggingFace 모델 다운로드")
        print("2: Civitai 모델 다운로드")
        print("3: 사용자 정의 저장소 관리")
        print("4: 설치된 모델 보기")
        print("5: 설정 관리")
        print("6: URL에서 직접 다운로드")
        print("7: 일괄 다운로드")
        print("8: 모델 무결성 검증")
        print("0: 종료")

        choice = input("\n선택: ")

        if choice == "1":
            print("\n사용 가능한 HuggingFace 모델:")
            for id, desc in manager.huggingface_descriptions.items():
                print(f"{id}: {desc}")

            model_ids = input("\n다운로드할 모델 ID (쉼표로 구분): ")
            if model_ids:
                for model_id in model_ids.split(","):
                    try:
                        manager.download_huggingface_model(
                            int(model_id.strip()))
                    except ValueError:
                        print(f"잘못된 모델 ID: {model_id}")

        elif choice == "2":
            url = input("Civitai 모델 URL: ")
            if url:
                manager.download_civitai_model(url)

        elif choice == "3":
            while True:
                print("\n=== 사용자 정의 저장소 관리 ===")
                print("1: 사용자 정의 저장소 추가")
                print("2: 사용자 정의 저장소 제거")
                print("3: 사용자 정의 저장소 나열")
                print("4: 사용자 정의 저장소에서 다운로드")
                print("0: 뒤로 가기")

                repo_choice = input("\n선택: ")

                if repo_choice == "1":
                    name = input("저장소 이름: ")
                    repo_id = input("HuggingFace Repo ID: ")
                    patterns = input("파일 패턴 (쉼표로 구분): ").split(",")

                    # 디렉토리 유형을 번호로 선택하도록 수정
                    print("디렉토리 유형을 선택하세요:")
                    dir_types = list(MODEL_DIRS.keys())
                    for idx, key in enumerate(dir_types, 1):
                        print(f"{idx}: {key}")

                    dir_choice = input("선택 (번호): ")
                    try:
                        dir_idx = int(dir_choice) - 1
                        if 0 <= dir_idx < len(dir_types):
                            dir_type = dir_types[dir_idx]
                            manager.add_custom_repo(
                                name, repo_id, patterns, dir_type)
                        else:
                            print("잘못된 선택입니다.")
                    except ValueError:
                        print("숫자를 입력해야 합니다.")

                elif repo_choice == "2":
                    name = input("제거할 저장소 이름: ")
                    manager.remove_custom_repo(name)

                elif repo_choice == "3":
                    print("\n사용자 정의 저장소:")
                    for name, repo in manager.config["custom_repos"].items():
                        print(f"- {name}:")
                        print(f"  Repo ID: {repo['repo_id']}")
                        print(f"  패턴: {repo['patterns']}")
                        print(f"  디렉토리: {repo['dir_type']}")

                elif repo_choice == "4":
                    # 먼저 저장소 목록을 표시
                    print("\n사용자 정의 저장소:")
                    repos = list(manager.config["custom_repos"].keys())
                    if not repos:
                        print("등록된 저장소가 없습니다.")
                        continue

                    for idx, name in enumerate(repos, 1):
                        repo = manager.config["custom_repos"][name]
                        print(
                            f"{idx}: {name} (Repo ID: {repo['repo_id']}, 디렉토리: {repo['dir_type']})")

                    print("\n팁: 여러 저장소를 선택하려면 다음 형식을 사용하세요:")
                    print("  - 범위 지정: '4-6' (4번부터 6번까지)")
                    print("  - 개별 지정: '4,6' (4번과 6번)")

                    repo_choice = input("\n다운로드할 저장소 선택 (번호, 범위 또는 이름): ")

                    # 단일 숫자로 입력한 경우
                    try:
                        idx = int(repo_choice) - 1
                        if 0 <= idx < len(repos):
                            name = repos[idx]
                            manager.download_custom_repo(name)
                        else:
                            print("잘못된 선택입니다.")
                    except ValueError:
                        # 범위 또는 콤마 구분 입력 확인
                        if "-" in repo_choice or "," in repo_choice:
                            manager.download_custom_repo(repo_choice)
                        # 이름으로 입력한 경우
                        elif repo_choice in repos:
                            manager.download_custom_repo(repo_choice)
                        else:
                            print(f"저장소 '{repo_choice}'를 찾을 수 없습니다.")

                elif repo_choice == "0":
                    break

        elif choice == "4":
            manager.list_models()

        elif choice == "5":
            while True:
                print("\n=== 설정 관리 ===")
                print("현재 설정:")
                for key, value in manager.config.items():
                    if key != "custom_repos":
                        print(f"- {key}: {value}")

                print("\n1: 기본 경로 변경")
                print("2: Civitai API 토큰 설정")
                print("3: 동시 다운로드 수 설정")
                print("4: 다운로드 재시도 횟수 설정")
                print("5: 다운로드 타임아웃 설정")
                print("0: 뒤로 가기")

                config_choice = input("\n선택: ")

                if config_choice == "1":
                    path = input("새 기본 경로: ")
                    manager.update_config("base_path", path)

                elif config_choice == "2":
                    token = input("Civitai API 토큰: ")
                    manager.update_config("civitai_token", token)

                elif config_choice == "3":
                    count = input("동시 다운로드 수: ")
                    manager.update_config("concurrent_downloads", count)

                elif config_choice == "4":
                    retries = input("다운로드 재시도 횟수: ")
                    manager.update_config("download_retries", retries)

                elif config_choice == "5":
                    timeout = input("다운로드 타임아웃 (초): ")
                    manager.update_config("download_timeout", timeout)

                elif config_choice == "0":
                    break

        elif choice == "6":
            url = input("다운로드할 URL: ")

            print("저장할 디렉토리 유형을 선택하세요:")
            dir_types = list(MODEL_DIRS.keys())
            for idx, key in enumerate(dir_types, 1):
                print(f"{idx}: {key}")

            dir_choice = input("선택 (번호): ")
            try:
                dir_idx = int(dir_choice) - 1
                if 0 <= dir_idx < len(dir_types):
                    dir_type = dir_types[dir_idx]
                    filename = input("저장할 파일명 (기본값: URL에서 추출): ")
                    if not filename:
                        filename = None
                    manager.direct_download(url, dir_type, filename)
                else:
                    print("잘못된 선택입니다.")
            except ValueError:
                print("숫자를 입력해야 합니다.")

        elif choice == "7":
            file_path = input("다운로드 목록 파일 경로: ")
            if not os.path.exists(file_path):
                print(f"파일을 찾을 수 없습니다: {file_path}")
                continue

            manager.batch_download(file_path)

        elif choice == "8":
            manager.verify_models()

        elif choice == "0":
            break


if __name__ == "__main__":
    main()
