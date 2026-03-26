import subprocess
import sys

def install_requirements():
    # 설치가 필요한 라이브러리 목록
    packages = [
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "matplotlib>=3.5.0"
    ]

    print("=== 🚀 드론 시뮬레이션 필수 라이브러리 자동 설치를 시작합니다 ===")
    
    for package in packages:
        print(f"\n📦 설치 및 확인 중: {package} ...")
        try:
            # 현재 실행 중인 파이썬 환경의 pip를 호출하여 설치를 진행합니다.
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ [{package}] 설치 완료 (또는 이미 최신 버전입니다)!")
        except subprocess.CalledProcessError:
            print(f"❌ [{package}] 설치에 실패했습니다. 에러 메시지를 확인해주세요.")

    print("\n🎉 모든 준비가 끝났습니다! 이제 드론 학습(train.py) 코드를 실행하셔도 좋습니다.")

if __name__ == "__main__":
    install_requirements()