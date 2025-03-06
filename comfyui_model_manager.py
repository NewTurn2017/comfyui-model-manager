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
from pathlib import Path
import shutil
from typing import List, Dict, Union, Optional, Any, Tuple
import concurrent.futures

# 기본 설정
DEFAULT_CONFIG = {
    "base_path": "/workspace/ComfyUI/models",  # 기본 모델 경로
    "civitai_token": "",  # Civitai API 토큰
    "custom_repos": {}  # 사용자 정의 저장소
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


class ComfyUIModelManager:
    def __init__(self):
        self.config = self._load_config()
        self.base_path = Path(self.config["base_path"])
        self.huggingface_models = {}
        self.huggingface_descriptions = {}
        self._load_huggingface_models()
        self._ensure_dependencies()
        self._ensure_directories_exist()

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
                             "hf_transfer>=0.1.8", "requests>=2.25.0"]

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
        from huggingface_hub import snapshot_download

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
            if patterns:
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=patterns,
                    local_dir=str(target_dir),
                    progress_bar=False
                )
            else:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target_dir),
                    progress_bar=False
                )
            print(f"다운로드 완료: {target_dir}")
            return True
        except Exception as e:
            print(f"다운로드 오류: {e}")
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

        return self._download_file(version_id, model_path, info['name'], info['model_name'])

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

    def _download_file(self, version_id: str, path: Path, filename: str, model_name: str) -> bool:
        """Civitai API를 사용하여 파일을 다운로드합니다."""
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / filename
        download_url = f"https://civitai.com/api/download/models/{version_id}"

        try:
            with requests.get(
                download_url,
                headers={
                    'Authorization': f'Bearer {self.config["civitai_token"]}'},
                stream=True
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                print(f"다운로드 중: {model_name} - {filename}")

                with open(file_path, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        f.write(chunk)
                        progress = (downloaded / total) * 100
                        print(
                            f"\r진행률: {progress:.1f}% ({downloaded}/{total} 바이트)", end='')

            print(f"\n저장 완료: {file_path}")
            return True
        except Exception as e:
            print(f"\n다운로드 실패: {e}")
            return False

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
                    progress_bar=False
                )
                print(f"다운로드 완료: {repo_name} → {target_dir}")
                return True
            except Exception as e:
                print(f"다운로드 오류 ({repo_name}): {e}")
                return False

        # 병렬로 다운로드 실행
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(repo_names))) as executor:
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
        for dir_type, dir_name in MODEL_DIRS.items():
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                files = list(dir_path.glob('*.*'))
                if files:
                    print(f"\n[{dir_type}] - {len(files)}개 파일:")
                    for file in files:
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print(f"  - {file.name} ({size_mb:.2f} MB)")

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

        # 설정 업데이트
        self.config[key] = value
        self._save_config()

        # base_path 변경 시 인스턴스 변수도 업데이트
        if key == "base_path":
            self.base_path = Path(value)
            self._ensure_directories_exist()

        print(f"설정이 업데이트되었습니다: {key} = {value}")
        return True


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

    elif args.command == "huggingface":
        if args.list:
            print("사용 가능한 HuggingFace 모델:")
            for id, desc in manager.huggingface_descriptions.items():
                repo_id = manager.huggingface_models[id][0]
                print(f"{id}: {desc} (repo: {repo_id})")
        elif args.download:
            manager.download_huggingface_model(args.download)

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

    elif args.command == "list":
        manager.list_models()

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

                    # 범위 또는 콤마 구분 입력 확인
                    if "-" in repo_choice or "," in repo_choice:
                        manager.download_custom_repo(repo_choice)
                    else:
                        try:
                            # 단일 번호로 입력한 경우
                            idx = int(repo_choice) - 1
                            if 0 <= idx < len(repos):
                                name = repos[idx]
                                manager.download_custom_repo(name)
                            else:
                                print("잘못된 선택입니다.")
                        except ValueError:
                            # 이름으로 입력한 경우
                            if repo_choice in repos:
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
                print("0: 뒤로 가기")

                config_choice = input("\n선택: ")

                if config_choice == "1":
                    path = input("새 기본 경로: ")
                    manager.update_config("base_path", path)

                elif config_choice == "2":
                    token = input("Civitai API 토큰: ")
                    manager.update_config("civitai_token", token)

                elif config_choice == "0":
                    break

        elif choice == "0":
            break


if __name__ == "__main__":
    main()
