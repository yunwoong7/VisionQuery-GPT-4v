<h2 align="center">
VisionQuery-GPT-4v
</h2>

<div align="center">
  <img src="https://img.shields.io/badge/python-v3.9.18-blue.svg"/>
  <img src="https://img.shields.io/badge/openai-v1.2.2-blue.svg"/>
</div>

AI의 발전은 끊임없이 우리의 기술적 지평을 확장시키고 있습니다. 특히, 이미지 인식 분야는 딥러닝의 발전과 함께 급속도로 진화해 왔습니다. 딥러닝 모델들은 이미지 인식, 객체 탐지, 얼굴 인식 등 다양한 분야에서 혁신적인 성과를 이루어냈지만 이러한 모델들은 주로 단일 모달리티, 이미지 데이터만을 처리하는 데 집중했습니다.

이번 글에서는 GPT-4V를 활용하여 딥러닝 모델에서 시도했던 다양한 이미지 인식 방법들을 재해석하고 적용해보려고 합니다. 구체적으로, Classification, Object Detection, Face Recognition, OCR 그리고 이미지 기반 추론 등의 주제를 다룰 예정입니다. 이를 통해 GPT-4V가 딥러닝 모델의 기능을 어떻게 향상할 수 있는지, 그리고 이미지와 텍스트를 결합하여 얻을 수 있는 새로운 인사이트와 가능성을 알아보도록 하겠습니다.

---

### #1. 필수 패키지 설치 및 준비

GPT-4V를 활용한 이미지 인식 프로젝트를 시작하기 전에, 필요한 Python 패키지들을 설치하고 준비하는 과정이 필요합니다.

**1) 패키지 설치
**먼저, 필요한 Python 라이브러리를 설치해야 합니다. 이를 위해 Python의 패키지 관리자인 pip를 사용합니다. 다음 명령어를 통해 필요한 패키지들을 설치할 수 있습니다.

```python
pip install openai requests pillow matplotlib
```

이 명령어는 OpenAI의 API를 사용하기 위한 openai, 웹에서 이미지를 다운로드하기 위한 requests, 이미지 처리를 위한 Pillow (PIL의 업데이트된 버전), 그리고 이미지를 시각화하기 위한 matplotlib를 설치합니다.

**2) 패키지 임포트
**설치가 완료된 후, Python 스크립트나 Jupyter 노트북에서 다음과 같이 필요한 패키지들을 임포트 합니다.

```python
import os
import openai
import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import base64
import io
```

- os: 운영 체제와 상호 작용하기 위한 모듈입니다. 파일 경로를 관리하고 환경 변수를 읽는 데 사용됩니다.
- openai: OpenAI의 API를 사용하기 위한 모듈입니다. GPT-4V와의 통신을 위해 필요합니다.
- requests: HTTP 요청을 보내기 위한 모듈입니다. 웹에서 이미지를 다운로드하는 데 사용됩니다.
- PIL (Pillow): 이미지 처리를 위한 라이브러리입니다. 이미지를 열고, 수정하고, 저장하는 데 사용됩니다.
- matplotlib.pyplot: 데이터를 시각화하기 위한 모듈입니다. 이미지를 표시하고 분석 결과를 시각화하는 데 사용됩니다.
- base64: 이미지를 base64 인코딩 형식으로 변환하는 데 사용됩니다. 이는 이미지를 GPT-4V API에 전송하기 위해 필요합니다.
- io: 파일 입출력을 위한 모듈입니다. 이미지를 메모리에서 직접 읽고 쓰는 데 사용됩니다.

### #2. OpenAI API 키 설정

GPT-4V를 사용하기 위해서는 OpenAI의 API 키를 설정하는 것이 필수적입니다.

**1) OpenAI API Key 발급**
OpenAI API 수행을 위해서는 먼저 [API Key 발급](https://platform.openai.com/)이 필요합니다. OpenAI 계정이 필요하며 계정이 없다면 계정 생성이 필요합니다. 간단히 Google이나 Microsoft 계정을 연동할 수 있습니다. 이미 계정이 있다면 로그인 후 진행하시면 됩니다.

![img](https://blog.kakaocdn.net/dn/c7Ud2A/btsAMoTcz9F/zkKjuobk0WjHMiAWQ6TYTk/img.png)

로그인이 되었다면 우측 상단 Personal -> [ View API Keys ]를 클릭합니다.

![img](https://blog.kakaocdn.net/dn/cXpfiv/btsAQAc9exp/nBczb5FcAAGyazumjKGxH0/img.png)

[ + Create new secret key ]를 클릭하여 API Key를 생성합니다. API key generated 창이 활성화되면 Key를 반드시 복사하여 두시기 바랍니다. 창을 닫으면 다시 확인할 수 없습니다. (만약 복사하지 못했다면 다시 Create new secret key 버튼을 눌러 생성하면 되니 걱정하지 않으셔도 됩니다.)

**2) 환경 변수 설정**
API 키를 직접 코드에 포함시키는 것은 보안상 좋지 않습니다. 대신, 운영 체제의 환경 변수에 API 키를 저장하고, 이를 코드에서 불러오는 방식을 권장합니다. 이를 위해 운영 체제의 환경 설정에서 OPENAI_API_KEY라는 이름으로 API 키를 저장합니다. 권장사항일 뿐 필수는 아닙니다.

**3) Open API 키 사용**
Python 코드에서는 os 모듈을 사용하여 환경 변수에서 API 키를 불러옵니다. 다음과 같이 코드를 작성합니다. 환경변수에 저장하지 않았다면 openai.api_key 값에 직접 입력하셔도 됩니다.

```python
openai.api_key = os.environ["OPENAI_API_KEY"]
client = openai.OpenAI()
```

### #3. Function Declaration

이미지 처리 및 GPT-4V 응답결과를 나타내기 위한 함수를 선언했는데 반복적인 내용을 줄이고 쉽게 사용하기 위한 목적입니다.

**1) 이미지 로드 및 인코딩 함수 (load_and_encode_images)**
이 함수는 이미지 소스(파일 경로 또는 URL)를 입력으로 받아, 해당 이미지를 PIL 이미지 객체로 변환하고, base64로 인코딩합니다. 이 과정은 GPT-4V에 이미지를 전송하기 위한 준비 단계입니다.

```python
# 이미지를 base64로 인코딩하고 PIL 이미지 객체를 반환하는 함수
def load_and_encode_images(image_sources):
    encoded_images = []
    pil_images = []
    for source in image_sources:
        if source.startswith('http'):  # URL인 경우
            response = requests.get(source)
            image_data = response.content
        else:  # 파일 경로인 경우
            with open(source, "rb") as image_file:
                image_data = image_file.read()

        pil_images.append(Image.open(io.BytesIO(image_data)))
        encoded_images.append(base64.b64encode(image_data).decode('utf-8'))
    return encoded_images, pil_images
```

**2) 응답결과 및 이미지를 출력하기 위한 함수 (display_response)**
이 함수는 PIL 이미지 객체와 GPT-4V의 응답 텍스트를 입력으로 받아, 이미지를 시각화하고 응답을 출력합니다.

```python
# 응답결과와 이미지를 출력하기 위한 함수
def display_response(pil_images, response_text):
    # 이미지 로딩 및 서브플롯 생성
    fig, axes = plt.subplots(nrows=1, ncols=len(pil_images), figsize=(5 * len(pil_images), 5))
    if len(pil_images) == 1:  # 하나의 이미지인 경우
        axes = [axes]

    # 이미지들 표시
    for i, img in enumerate(pil_images):
        axes[i].imshow(img)
        axes[i].axis('off')  # 축 정보 숨기기
        axes[i].set_title(f'Image #{i+1}')

    # 전체 플롯 표시
    plt.show()

    print(response_text)
```

**3) 이미지 처리 및 GPT-4V 요청 함수 (process_and_display_images)**
이 함수는 이미지 소스와 사용자의 프롬프트를 입력으로 받아, 이미지를 처리하고 GPT-4V에 요청을 보낸 후, 응답과 이미지를 표시합니다.

```python
# 이미지 경로 또는 URL과 프롬프트를 처리하는 함수
def process_and_display_images(image_sources, prompt):
    # 이미지 로드, base64 인코딩 및 PIL 이미지 객체 생성
    base64_images, pil_images = load_and_encode_images(image_sources)

    # OpenAI에 요청 보내기
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=1000
    )

    response_text = response.choices[0].message.content

    # 응답과 이미지 표시
    display_response(pil_images, response.choices[0].message.content)

    return response_text
```

---

### #4. Classification

이미지 분류는 GPT-4V를 활용하여 이미지 내의 객체를 특정 카테고리로 분류하는 과정입니다. 이를 통해 이미지가 어떤 객체를 포함하고 있는지를 정확하게 식별할 수 있습니다.

```python
# 예시: 이미지 경로 리스트
image_paths = ["asset/images/test_8.png", "asset/images/test_9.png"]  # N개의 이미지 경로
classes = ["벤츠", "기아", "현대"]
prompt = "이미지를 분석하여 이미지에 있는 객체의 클래스를 반환합니다. 해당 목록에서는 하나의 클래스만 반환할 수 있습니다. 클래스는 다음과 같습니다: {}".format(classes)
response_text = process_and_display_images(image_paths, prompt)
```

![img](https://blog.kakaocdn.net/dn/nlu4I/btsAOLNCN8S/L8TUjmweC0q9IRHDa1l9iK/img.png)

```python
image_paths = ["asset/images/test_10.png"]  # N개의 이미지 경로
prompt = "아우디 차량만 찾아서 번호를 알려줘"
response_text = process_and_display_images(image_paths, prompt)
```

![img](https://blog.kakaocdn.net/dn/8SVWA/btsAPxHYLjk/RWZoPdm6SnJ6vtY4m7BakK/img.png)

### #5. Object Detection

```python
image_sources = ["asset/images/test_1.png"]
prompt = "Can you tell me the location of the dog on the image, very accurately, ensuring that the area covers the entire object (dog). Share the x_min, y_min, x_max, y_max in 0-1 normalized space. Only return the numbers, nothing else."
response_text = process_and_display_images(image_sources, prompt)
```

![img](https://blog.kakaocdn.net/dn/b46qJe/btsASukpAn4/t5f9Lop73j66KR0EJnooQk/img.png)

Object Detection 결과는 기대했던 수준에 못 미치는 것 같습니다. 그리고 여러 번 수행해 봤지만 응답결과의 대부분은 *"Sorry, I can't assist with that request."* 형태로 처리가 불가하다는 메시지로 반환됩니다.

### #6. OCR

```python
image_sources = ["asset/images/test_3.png"]
prompt = "이 사진에서 텍스트 추출해서 OCR 수행해줘"
response_text = process_and_display_images(image_sources, prompt)
```

![img](https://blog.kakaocdn.net/dn/OChVW/btsASra7X2U/B43xNN9J8sTuJvaniFD320/img.png)

### #7. Face Recognition

```python
# 이미지 경로 또는 URL 리스트
image_sources = ["asset/images/test_5.png", "asset/images/test_7.png"]
prompt = "첫 번째 이미지의 사람이 두 번째 이미지에 몇 번에 해당하는지 찾아서 번호만 알려주세요."
response_text = process_and_display_images(image_sources, prompt)
```

![img](https://blog.kakaocdn.net/dn/APoS7/btsATY7bJVF/DigCxk8kXlX007eB1k6KIk/img.png)

### #8. 이미지 기반 추론

```python
# 예시: 이미지 경로 리스트
image_paths = ["asset/images/test_11.png"]  # N개의 이미지 경로
prompt = "이미지에서 자를 기반으로 펜의 크기를 추론하여 Cm로 알려줘"
response_text = process_and_display_images(image_paths, prompt)
```

![img](https://blog.kakaocdn.net/dn/ec9PYX/btsBlcJQ238/a9HKu155X8iSZsyRH68OEK/img.png)

![img](https://blog.kakaocdn.net/dn/RAEq8/btsBjP2vF0M/dFHFbikbexH2RTkmcWQs1k/img.png)
