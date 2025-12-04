"""
画像生成AIアプリ プロトタイプ
Streamlit + Google Gemini API

有料版向け:
- Gemini 2.5 Flash: 画像分析・プロンプト作成
- Imagen 4.0: 高品質画像生成
"""

import os
import io
import hashlib
from datetime import datetime
from PIL import Image, ExifTags
import streamlit as st
from dotenv import load_dotenv
from google import genai as genai_client
from google.genai import types as genai_types

# .envファイルから環境変数を読み込む（ローカル開発用）
load_dotenv()

# デフォルトAPIキー（Streamlit Cloudのシークレットまたは環境変数から取得）
def get_default_api_key():
    """APIキーを取得（優先順位: Streamlit secrets > 環境変数 > 空）"""
    # Streamlit Cloudのシークレットから取得
    try:
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            return st.secrets['GOOGLE_API_KEY']
    except Exception:
        pass
    # 環境変数から取得
    return os.getenv("GOOGLE_API_KEY", "")

# モデル設定
MODELS = {
    "reasoning": "gemini-2.5-flash",           # 画像分析・プロンプト作成
    "gemini_image": "nano-banana-pro-preview", # Gemini画像生成（日本語対応）
    "imagen": "imagen-4.0-generate-001",       # Imagen画像生成（写真風）
}

def get_client(api_key: str) -> genai_client.Client:
    """Google GenAI クライアントを作成"""
    return genai_client.Client(api_key=api_key)


def compute_cache_key(
    images: list[Image.Image],
    prompt: str
) -> str:
    """入力からキャッシュキーを生成"""
    hasher = hashlib.md5()
    hasher.update(prompt.encode())
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        hasher.update(buf.getvalue())
    return hasher.hexdigest()


def fix_image_orientation(image: Image.Image) -> Image.Image:
    """EXIF情報に基づいて画像の向きを修正する（スマホ縦長画像対応）"""
    try:
        exif = image._getexif()
        if exif is None:
            return image

        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == 'Orientation':
                orientation_key = key
                break

        if orientation_key is None or orientation_key not in exif:
            return image

        orientation = exif[orientation_key]

        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 7:
            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)

        return image
    except Exception:
        return image


def load_image_as_pil(uploaded_file) -> Image.Image:
    """StreamlitのUploadedFileをPIL Imageに変換（EXIF向き修正付き）"""
    image = Image.open(uploaded_file)
    image = fix_image_orientation(image)
    return image


def pil_to_part(image: Image.Image) -> genai_types.Part:
    """PIL ImageをGenAI Part形式に変換"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return genai_types.Part.from_bytes(
        data=buf.read(),
        mime_type="image/png"
    )


def analyze_and_create_prompt(
    client: genai_client.Client,
    images: list[Image.Image],
    user_prompt: str
) -> str:
    """
    Geminiで画像を分析し、画像生成用の詳細なプロンプトを作成する
    """
    analysis_prompt = f"""あなたは画像分析と画像生成プロンプト作成の専門家です。

以下の画像を詳細に分析し、ユーザーの指示に基づいて、
画像生成AI（Imagen 4.0）に渡すための詳細な英語プロンプトを作成してください。

【ユーザーの指示】
{user_prompt}

【重要】
- ユーザーの曖昧な指示（「もっとカッコよく」「いい感じに」など）を、
  具体的な画像生成プロンプトに変換してください
- 画像の主要な要素（被写体、色、構図、スタイル、雰囲気）を把握し、
  ユーザーの指示を反映した新しい画像の詳細な描写を作成してください
- 人物が写っている場合は、その人物の特徴も詳しく描写してください

【出力形式】
以下の形式で出力してください：

## 画像分析
（各画像の分析結果を日本語で簡潔に）

## ユーザー意図の解釈
（ユーザーの指示をどう解釈したか日本語で説明）

## 生成プロンプト
（Imagen 4.0用の詳細な英語プロンプト。1段落で、具体的な描写を含む。
スタイル、構図、色調、雰囲気なども明確に指定する）
"""

    image_parts = [pil_to_part(img) for img in images]
    contents = [analysis_prompt] + image_parts

    response = client.models.generate_content(
        model=MODELS["reasoning"],
        contents=contents
    )

    # レスポンスからテキストを取得
    if response.text:
        return response.text

    # response.textがNoneの場合、candidatesから取得を試みる
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                return part.text

    raise ValueError("画像分析の結果を取得できませんでした")


def extract_imagen_prompt(analysis_text: str) -> str:
    """分析結果から英語プロンプト部分を抽出"""
    if not analysis_text:
        raise ValueError("分析テキストが空です")

    if "## 生成プロンプト" in analysis_text:
        prompt_section = analysis_text.split("## 生成プロンプト")[1]
        # 余分な改行やマークダウン記号を削除
        prompt = prompt_section.strip()
        # 次のセクションがあれば切り取る
        if "##" in prompt:
            prompt = prompt.split("##")[0].strip()
        return prompt
    return analysis_text


def generate_with_gemini(
    client: genai_client.Client,
    prompt: str,
    images: list[Image.Image] = None
) -> Image.Image | None:
    """Geminiで画像を生成（思考モード有効・日本語テキスト対応）"""
    # コンテンツを構築
    if images:
        image_parts = [pil_to_part(img) for img in images]
        contents = [prompt] + image_parts
    else:
        contents = prompt

    response = client.models.generate_content(
        model=MODELS["gemini_image"],
        contents=contents,
        config=genai_types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            thinking_config=genai_types.ThinkingConfig(
                thinking_budget=2048  # 思考トークン数
            )
        )
    )

    # レスポンスから画像を抽出
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                image_data = part.inline_data.data
                return Image.open(io.BytesIO(image_data))

    return None


def generate_with_gemini_thinking(
    client: genai_client.Client,
    prompt: str,
    images: list[Image.Image]
) -> Image.Image | None:
    """Geminiで画像を生成（思考モード有効・画像入力必須）"""
    # コンテンツを構築
    image_parts = [pil_to_part(img) for img in images]
    contents = [prompt] + image_parts

    response = client.models.generate_content(
        model=MODELS["gemini_image"],
        contents=contents,
        config=genai_types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            thinking_config=genai_types.ThinkingConfig(
                thinking_budget=2048  # 思考トークン数
            )
        )
    )

    # レスポンスから画像を抽出
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                image_data = part.inline_data.data
                return Image.open(io.BytesIO(image_data))

    return None


def generate_with_imagen4(
    client: genai_client.Client,
    prompt: str
) -> Image.Image | None:
    """Imagen 4.0で画像を生成（写真風）"""
    response = client.models.generate_images(
        model=MODELS["imagen"],
        prompt=prompt,
        config=genai_types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="BLOCK_LOW_AND_ABOVE",
            person_generation="ALLOW_ADULT",
        )
    )

    if response.generated_images:
        image_data = response.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(image_data))

    return None


def init_session_state():
    """セッションステートを初期化"""
    # テキスト生成モード用
    if "text_generated_image" not in st.session_state:
        st.session_state.text_generated_image = None
    if "text_generation_complete" not in st.session_state:
        st.session_state.text_generation_complete = False
    if "text_prompt" not in st.session_state:
        st.session_state.text_prompt = ""
    # 画像加工モード用
    if "image_generated_image" not in st.session_state:
        st.session_state.image_generated_image = None
    if "image_generation_complete" not in st.session_state:
        st.session_state.image_generation_complete = False
    if "image_prompt" not in st.session_state:
        st.session_state.image_prompt = ""
    # ファイルアップローダーのキー用カウンター（クリア時にインクリメント）
    if "uploader_key_counter" not in st.session_state:
        st.session_state.uploader_key_counter = 0
    # 共通
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "📝 テキストから生成"
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "pending_images" not in st.session_state:
        st.session_state.pending_images = None
    if "pending_mode" not in st.session_state:
        st.session_state.pending_mode = None
    # 履歴機能用（LocalStorageからの読み込みはmainで行う）
    if "image_history" not in st.session_state:
        st.session_state.image_history = []  # [{"image": PIL.Image, "prompt": str, "timestamp": str}, ...]
    if "history_loaded" not in st.session_state:
        st.session_state.history_loaded = False


def main():
    st.set_page_config(
        page_title="画像生成AI",
        page_icon="🎨",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # カスタムCSS（余白を詰める・ヘッダー非表示）
    st.markdown("""
    <style>
        /* ヘッダー（Deploy等）を非表示 */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        /* メインメニューボタンを非表示 */
        #MainMenu {
            display: none !important;
        }
        /* フッターを非表示 */
        footer {
            display: none !important;
        }
        /* メインコンテンツの上部余白を削減 */
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        /* タイトルの余白を削減 */
        h1 {
            margin-top: 0;
            margin-bottom: 0.3rem;
        }
        /* 区切り線の余白を削減 */
        hr {
            margin-top: 0.3rem;
            margin-bottom: 0.3rem;
        }
        /* ラジオボタンの余白を削減 */
        .stRadio > div {
            margin-bottom: 0;
        }
        .stRadio > label {
            margin-bottom: 0;
        }
        /* テキストエリアの余白を削減 */
        .stTextArea > div {
            margin-bottom: 0.3rem;
        }
        .stTextArea label {
            margin-bottom: 0.2rem;
        }
        /* ボタンの余白を削減 */
        .stButton {
            margin-top: 0.3rem;
        }
        /* サブヘッダーの余白を削減 */
        h3 {
            margin-top: 0;
            margin-bottom: 0.2rem;
        }
        /* ファイルアップローダーの余白を削減 */
        .stFileUploader {
            margin-bottom: 0.3rem;
        }
        /* 段落の余白を削減 */
        p {
            margin-bottom: 0.3rem;
        }
        /* カラムの余白を削減 */
        [data-testid="column"] {
            padding: 0 0.3rem;
        }
        /* 成功メッセージの余白 */
        .stSuccess, .stError, .stInfo {
            margin-top: 0.3rem;
            margin-bottom: 0.3rem;
        }
        /* ローディングオーバーレイ */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(255, 255, 255, 0.9);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #ff6b6b;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-text {
            margin-top: 20px;
            font-size: 1.2rem;
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True)

    # セッション初期化
    init_session_state()

    # 生成中はオーバーレイを表示して処理を実行
    if st.session_state.is_generating:
        st.markdown("""
        <div class="loading-overlay">
            <div class="loading-spinner"></div>
            <div class="loading-text">🧠 思考モードで画像を生成中...</div>
        </div>
        """, unsafe_allow_html=True)

        # APIキーを取得
        default_api_key = get_default_api_key()
        api_key = st.session_state.get("api_key", default_api_key)
        client = get_client(api_key)

        try:
            prompt = st.session_state.pending_prompt
            images = st.session_state.pending_images
            pending_mode = st.session_state.pending_mode

            if images:
                # 画像加工モード
                generated_image = generate_with_gemini_thinking(client, prompt, images)
                st.session_state.image_generated_image = generated_image
                st.session_state.image_generation_complete = True
            else:
                # テキスト生成モード
                generated_image = generate_with_gemini(client, prompt)
                st.session_state.text_generated_image = generated_image
                st.session_state.text_generation_complete = True

            # 履歴に追加（生成成功時のみ）
            if generated_image is not None:
                st.session_state.image_history.insert(0, {
                    "image": generated_image,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                # 最大枚数を超えたら古いものを削除（20枚まで）
                if len(st.session_state.image_history) > 20:
                    st.session_state.image_history = st.session_state.image_history[:20]

            # 生成完了後、元のモードを維持
            st.session_state.current_mode = pending_mode

        except Exception as e:
            st.session_state.generation_error = str(e)

        finally:
            st.session_state.is_generating = False
            st.session_state.pending_prompt = None
            st.session_state.pending_images = None
            st.session_state.pending_mode = None
            st.rerun()

    st.title("🎨 画像生成AIアプリ")

    # モード選択（セッションステートで管理）
    mode_options = ["📝 テキストから生成", "🖼️ 画像を加工・合成"]
    current_index = mode_options.index(st.session_state.current_mode) if st.session_state.current_mode in mode_options else 0

    mode = st.radio(
        "生成モードを選択",
        mode_options,
        index=current_index,
        horizontal=True,
        key="mode_selector"
    )

    # モードの変更を保存
    st.session_state.current_mode = mode

    # APIキーはセッションステートで管理
    default_api_key = get_default_api_key()
    if "api_key" not in st.session_state:
        st.session_state.api_key = default_api_key
    api_key = st.session_state.api_key

    # --- メインエリア ---

    if mode == "📝 テキストから生成":
        # テキストから生成モード
        st.write("プロンプトを入力して、AIで画像を生成します")

        # on_changeでセッションステートに保存（モード切替時も値を保持）
        def save_text_prompt():
            st.session_state.text_prompt_saved = st.session_state.text_prompt_widget

        # 保存された値があれば使用
        default_text = st.session_state.get("text_prompt_saved", "")

        prompt = st.text_area(
            "生成したい画像を説明してください",
            value=default_text,
            placeholder="例: 夕焼けの海辺で遊ぶ柴犬 / 未来都市の夜景 / 森の中の小さなコテージ",
            height=400,
            key="text_prompt_widget",
            on_change=save_text_prompt
        )
        # 現在の値も保存
        st.session_state.text_prompt_saved = prompt

        # ボタン行（生成 + クリア）
        col_gen, col_clear = st.columns([3, 1])
        with col_gen:
            generate_clicked = st.button(
                "🚀 生成する",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_generating
            )
        with col_clear:
            if st.button("🗑️ クリア", use_container_width=True, disabled=st.session_state.is_generating):
                st.session_state.text_generated_image = None
                st.session_state.text_generation_complete = False
                # 保存された値をクリア
                st.session_state.text_prompt_saved = ""
                if "text_prompt_widget" in st.session_state:
                    del st.session_state["text_prompt_widget"]
                st.rerun()

        # 注意文
        st.caption("⚠️ 短時間に連続して生成するとAPI制限に引っかかる場合があります")

        # 生成ボタン
        if generate_clicked:
            # バリデーション
            if not api_key:
                st.error("❌ APIキーを設定してください")
            elif not prompt:
                st.error("❌ プロンプトを入力してください")
            else:
                # 前回の生成画像をクリア & 生成開始
                st.session_state.text_generated_image = None
                st.session_state.text_generation_complete = False
                st.session_state.generation_error = None
                st.session_state.pending_prompt = prompt
                st.session_state.pending_images = None
                st.session_state.pending_mode = "📝 テキストから生成"
                st.session_state.is_generating = True
                st.rerun()

    else:
        # 画像加工・合成モード
        st.write("画像をアップロードして、AIで加工・合成しましょう")

        # 画像アップロード（2列レイアウト）
        col1, col2 = st.columns(2)
        # キーにカウンターを追加（クリア時に新しいウィジェットとして認識させる）
        uploader_counter = st.session_state.uploader_key_counter

        with col1:
            st.subheader("画像1")
            uploaded_image1 = st.file_uploader(
                "画像1をアップロード",
                type=["jpg", "jpeg", "png"],
                key=f"image1_{uploader_counter}",
                label_visibility="collapsed"
            )
            # 新しいファイルがアップロードされた場合はキャッシュを更新
            if uploaded_image1:
                file_id1 = f"{uploaded_image1.name}_{uploaded_image1.size}"
                if st.session_state.get("preview1_file_id") != file_id1:
                    uploaded_image1.seek(0)
                    preview1 = load_image_as_pil(uploaded_image1)
                    st.session_state.preview1_cache = preview1
                    st.session_state.preview1_file_id = file_id1
            # キャッシュがあれば常に表示（ファイルアップローダーの状態に関係なく）
            if st.session_state.get("preview1_cache") is not None:
                st.image(st.session_state.preview1_cache, caption="画像1 プレビュー", width=200)

        with col2:
            st.subheader("画像2")
            uploaded_image2 = st.file_uploader(
                "画像2をアップロード",
                type=["jpg", "jpeg", "png"],
                key=f"image2_{uploader_counter}",
                label_visibility="collapsed"
            )
            # 新しいファイルがアップロードされた場合はキャッシュを更新
            if uploaded_image2:
                file_id2 = f"{uploaded_image2.name}_{uploaded_image2.size}"
                if st.session_state.get("preview2_file_id") != file_id2:
                    uploaded_image2.seek(0)
                    preview2 = load_image_as_pil(uploaded_image2)
                    st.session_state.preview2_cache = preview2
                    st.session_state.preview2_file_id = file_id2
            # キャッシュがあれば常に表示（ファイルアップローダーの状態に関係なく）
            if st.session_state.get("preview2_cache") is not None:
                st.image(st.session_state.preview2_cache, caption="画像2 プレビュー", width=200)

        st.divider()

        # on_changeでセッションステートに保存（モード切替時も値を保持）
        def save_image_prompt():
            st.session_state.image_prompt_saved = st.session_state.image_prompt_widget

        # 保存された値があれば使用
        default_image_text = st.session_state.get("image_prompt_saved", "")

        prompt = st.text_area(
            "どのように加工・合成しますか？",
            value=default_image_text,
            placeholder="例: もっとカッコよくして / 2つの画像を自然に合成して夕焼けの背景にして / いい感じにして",
            height=400,
            key="image_prompt_widget",
            on_change=save_image_prompt
        )
        # 現在の値も保存
        st.session_state.image_prompt_saved = prompt

        # ボタン行（生成 + クリア）
        col_gen2, col_clear2 = st.columns([3, 1])
        with col_gen2:
            generate_clicked2 = st.button(
                "🚀 生成する",
                type="primary",
                use_container_width=True,
                key="gen2",
                disabled=st.session_state.is_generating
            )
        with col_clear2:
            if st.button("🗑️ クリア", use_container_width=True, key="clear2", disabled=st.session_state.is_generating):
                st.session_state.image_generated_image = None
                st.session_state.image_generation_complete = False
                # 保存された値をクリア
                st.session_state.image_prompt_saved = ""
                if "image_prompt_widget" in st.session_state:
                    del st.session_state["image_prompt_widget"]
                # ファイルアップローダーをリセット（カウンターをインクリメント）
                st.session_state.uploader_key_counter += 1
                # プレビューキャッシュもクリア
                st.session_state.preview1_cache = None
                st.session_state.preview2_cache = None
                st.session_state.preview1_file_id = None
                st.session_state.preview2_file_id = None
                st.rerun()

        # 注意文
        st.caption("⚠️ 短時間に連続して生成するとAPI制限に引っかかる場合があります")

        # 生成ボタン
        if generate_clicked2:
            # バリデーション
            if not api_key:
                st.error("❌ APIキーを設定してください")
            elif not uploaded_image1 and not uploaded_image2:
                st.error("❌ 少なくとも1つの画像をアップロードしてください")
            elif not prompt:
                st.error("❌ 加工指示を入力してください")
            else:
                # 画像をPIL形式に変換
                images = []
                if uploaded_image1:
                    uploaded_image1.seek(0)
                    images.append(load_image_as_pil(uploaded_image1))
                if uploaded_image2:
                    uploaded_image2.seek(0)
                    images.append(load_image_as_pil(uploaded_image2))

                # 前回の生成画像をクリア & 生成開始
                st.session_state.image_generated_image = None
                st.session_state.image_generation_complete = False
                st.session_state.generation_error = None
                st.session_state.pending_prompt = prompt
                st.session_state.pending_images = images
                st.session_state.pending_mode = "🖼️ 画像を加工・合成"
                st.session_state.is_generating = True
                st.rerun()

    # --- エラー表示 ---
    if st.session_state.get("generation_error"):
        st.error(f"❌ エラーが発生しました: {st.session_state.generation_error}")
        st.session_state.generation_error = None

    # --- 結果表示（モードごとに別々） ---
    if mode == "📝 テキストから生成":
        # テキスト生成モードの結果
        if st.session_state.text_generation_complete:
            if st.session_state.text_generated_image:
                st.success("✅ 画像の生成が完了しました！")
                st.image(
                    st.session_state.text_generated_image,
                    caption="生成された画像",
                    use_container_width=True
                )

                buf = io.BytesIO()
                st.session_state.text_generated_image.save(buf, format="PNG")
                st.download_button(
                    label="📥 画像をダウンロード",
                    data=buf.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png"
                )
            else:
                st.error("❌ 画像の生成に失敗しました。別のプロンプトを試してください。")
    else:
        # 画像加工モードの結果
        if st.session_state.image_generation_complete:
            if st.session_state.image_generated_image:
                st.success("✅ 画像の生成が完了しました！")
                st.image(
                    st.session_state.image_generated_image,
                    caption="生成された画像",
                    use_container_width=True
                )

                buf = io.BytesIO()
                st.session_state.image_generated_image.save(buf, format="PNG")
                st.download_button(
                    label="📥 画像をダウンロード",
                    data=buf.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png",
                    key="download2"
                )
            else:
                st.error("❌ 画像の生成に失敗しました。別のプロンプトを試してください。")

    # --- 履歴セクション ---
    if st.session_state.image_history:
        st.divider()
        st.subheader("📜 生成履歴")

        # 履歴をグリッド表示（3列）
        cols_per_row = 3
        for i in range(0, len(st.session_state.image_history), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(st.session_state.image_history):
                    item = st.session_state.image_history[idx]
                    with col:
                        st.image(item["image"], use_container_width=True)
                        st.caption(f"🕐 {item['timestamp']}")
                        st.caption(f"📝 {item['prompt'][:30]}..." if len(item['prompt']) > 30 else f"📝 {item['prompt']}")

                        # ダウンロードと削除ボタン
                        col_dl, col_del = st.columns(2)
                        with col_dl:
                            buf = io.BytesIO()
                            item["image"].save(buf, format="PNG")
                            st.download_button(
                                label="📥",
                                data=buf.getvalue(),
                                file_name=f"image_{idx}.png",
                                mime="image/png",
                                key=f"dl_hist_{idx}",
                                use_container_width=True
                            )
                        with col_del:
                            if st.button("🗑️", key=f"del_hist_{idx}", use_container_width=True):
                                st.session_state.image_history.pop(idx)
                                st.rerun()

        # 履歴全削除ボタン
        st.divider()
        if st.button("🗑️ 履歴を全て削除", type="secondary"):
            st.session_state.image_history = []
            st.rerun()


if __name__ == "__main__":
    main()
