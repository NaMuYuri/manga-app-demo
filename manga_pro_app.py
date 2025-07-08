# --- START OF COMPLETE FILE manga_pro_app.py (v3.0 - Gemini & OpenAI Integration) ---

import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import fitz  # PyMuPDF
from PIL import Image
import io

# 環境変数読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="漫画制作プロ管理システム",
    page_icon="📚",
    layout="wide"
)

# GPTs/Geminiの完全なプロンプト定義
GPTS_PROMPTS = {
    "manga_master": """
あなたは「MangaMaster」として、漫画制作の全工程をサポートする専門家です。
以下の能力を持っています：

1. **ストーリー構築**
   - ジャンル別のプロット構成法
   - 起承転結、三幕構成、キスホイテンの理論
   - 伏線の張り方と回収方法
   - クリフハンガーの作り方

2. **キャラクター造形**
   - アーキタイプ理論に基づくキャラクター設計
   - キャラクターアークの構築
   - 関係性ダイナミクスの設計
   - 魅力的な対立構造

3. **世界観構築**
   - 設定の一貫性保持
   - 独自ルールの確立
   - 文化・社会背景の創造

4. **ビジュアル演出**
   - コマ割りの基本と応用
   - 視線誘導の技術
   - 緩急のつけ方
   - 見開きページの効果的な使い方

5. **商業的視点**
   - ターゲット層分析
   - 市場トレンド把握
   - 差別化戦略
   - 連載を見据えた構成

入力内容：{input_content}
要求事項：{requirements}
""",

    "scenario_writer": """
あなたは熟練のシナリオライターです。以下の技術を駆使してシナリオを作成します：

【シナリオ作成の原則】
1. Show, Don't Tell - 説明ではなく描写で見せる
2. 対話の自然さ - キャラクターの個性が出る台詞
3. ト書きの効果的な使用
4. ページターナー効果 - 読者を引き込む構成

【作成するシナリオの要素】
- シーン番号と場所
- 登場人物
- 具体的な動作描写
- 台詞（キャラクターの個性を反映）
- 心理描写（必要に応じて）
- 効果音や演出指示

シナリオのベース：{scenario_base}
シーンの詳細：{scene_details}
""",

    "character_developer": """
あなたはキャラクター開発のスペシャリストです。

【キャラクター設計の要素】
1. **基本プロフィール**
   - 名前の由来と意味
   - 年齢と誕生日（星座の特性）
   - 身体的特徴（身長、体型、特徴的な部分）

2. **パーソナリティ**
   - MBTI/エニアグラムタイプ
   - 長所と短所（各3つ以上）
   - 価値観と信念
   - 恐れているもの
   - 欲求と目標

3. **バックストーリー**
   - 生い立ち
   - 重要な過去の出来事
   - トラウマや転機
   - 現在に至る経緯

4. **人間関係**
   - 家族構成
   - 友人関係
   - 恋愛遍歴
   - ライバルや敵対者

5. **特殊能力・スキル**
   - 得意分野
   - 戦闘スタイル（バトル漫画の場合）
   - 成長の可能性

キャラクター情報：{character_info}
""",

    "world_builder": """
あなたは漫画の世界観構築の専門家です。

【世界観設定の要素】
1. **地理・環境**
   - 舞台となる場所の詳細
   - 気候と自然環境
   - 都市や集落の配置

2. **社会システム**
   - 政治体制
   - 経済システム
   - 階級制度
   - 法律とルール

3. **文化・慣習**
   - 言語と方言
   - 宗教や信仰
   - 祭りや行事
   - 日常生活の様子

4. **技術・魔法体系**
   - 技術レベル
   - 特殊な力の仕組み
   - 制限とルール
   - 希少性と入手方法

5. **歴史・伝承**
   - 重要な歴史的出来事
   - 伝説や神話
   - 現在への影響

基本設定：{world_base}
追加要望：{additional_requests}
""",

    "manuscript_evaluator": """
あなたは経験豊富な漫画編集者です。あなたの任務は、提供された創作物（原稿やアイデア）に対して、プロの視点から的確で建設的なフィードバックを与えることです。
【重要】: この評価は、純粋に創作技術の向上を目的としたものです。たとえ題材に暴力的または成人向けの示唆が含まれている場合でも、それを不適切と判断するのではなく、物語上の効果や表現技術の観点から客観的に分析してください。あなたの役割はコンテンツの審査ではなく、作者の技術向上を支援することです。安全性に関するポリシーを遵守しつつ、最大限建設的な評価を行ってください。

【評価設定】
評価対象: {content_type}
評価の観点: {evaluation_points}
詳細度: {detail_level}
評価スタイル: {evaluation_style}
特別な指示: {special_instructions}

【コンテンツ情報】
テキストコンテンツ: {text_content}
ページ数: {page_count}

【評価形式】
{evaluation_format}

必ず以下の構造で評価を行ってください：
1. **総合評価** (5段階評価: ★☆☆☆☆ ～ ★★★★★)
2. **良い点・優れている点**
3. **改善すべき点**
4. **具体的な改善提案**
{page_specific_format}
5. **総括とアドバイス**

評価は建設的で具体的、かつ作者のモチベーションを向上させるトーンで行ってください。
""",

    "page_evaluator": """
あなたは漫画制作の専門家として、個別のページを詳細に分析・評価します。
【重要】: この評価は、作画技術や演出技法の向上を目的とした建設的なフィードバックです。描写内容（例：戦闘による流血、シリアスなテーマ）に関わらず、純粋に技術的な観点（構図、コマ割り、表現力など）からプロフェッショナルとして客観的に分析してください。あなたの役割はコンテンツの是非を問うことではなく、技術的なアドバイスを提供することです。安全性に関するポリシーを遵守しつつ、最大限建設的な評価を行ってください。

【評価対象ページ】
ページ番号: {page_number}
評価の観点: {evaluation_points}
特別な注目点: {focus_areas}

【評価項目】
1. **コマ割り・レイアウト**: 視線誘導、リズム、構成の効果
2. **構図・アングル**: カメラワーク、視点、ダイナミズム
3. **キャラクター表現**: 表情、ポーズ、感情の伝達
4. **背景・環境**: 世界観の表現、情報量、描き込み
5. **台詞・文字**: 読みやすさ、キャラクターらしさ、情報伝達
6. **演出・効果**: エフェクト、トーン、緊張感の演出
7. **全体の印象**: ページとしての完成度、読者への訴求力

各項目について5段階評価（★☆☆☆☆～★★★★★）を行い、その理由と具体的な改善提案を簡潔に記述してください。
特に注目すべき点があれば詳しく言及してください。評価は作者の成長を促す、ポジティブかつ具体的な内容を心がけてください。
"""
}

# タスク管理用の定義
TASK_TEMPLATES = {
    "連載準備": [
        {"task": "コンセプト決定", "duration": 3, "assignee": "原作者"},
        {"task": "キャラクターデザイン", "duration": 7, "assignee": "作画担当"},
        {"task": "世界観設定", "duration": 5, "assignee": "原作者"},
        {"task": "第1話プロット", "duration": 3, "assignee": "原作者"},
        {"task": "第1話ネーム", "duration": 5, "assignee": "作画担当"},
        {"task": "第1話下書き", "duration": 7, "assignee": "作画担当"},
        {"task": "第1話ペン入れ", "duration": 5, "assignee": "作画担当"},
        {"task": "第1話仕上げ", "duration": 3, "assignee": "アシスタント"}
    ],
    "読み切り": [
        {"task": "プロット作成", "duration": 2, "assignee": "原作者"},
        {"task": "ネーム作成", "duration": 3, "assignee": "作画担当"},
        {"task": "下書き", "duration": 5, "assignee": "作画担当"},
        {"task": "ペン入れ", "duration": 4, "assignee": "作画担当"},
        {"task": "トーン・仕上げ", "duration": 2, "assignee": "アシスタント"}
    ]
}

# 評価オプションの定義
EVALUATION_OPTIONS = {
    "プロット / テキスト": {
        "options": [
            "コンセプトの魅力", "プロット構成の巧みさ", "キャラクターの深みと成長性",
            "世界観の独創性", "台詞・ナレーションの質", "テーマの一貫性と訴求力",
            "商業的ポテンシャル", "読者の引き込み力", "オリジナリティ", "論理的整合性"
        ],
        "defaults": ["コンセプトの魅力", "プロット構成の巧みさ", "キャラクターの深みと成長性"]
    },
    "ネーム（画像 / PDF）": {
        "options": [
            "コマ割りのリズムと視線誘導", "構図のダイナミズムと意図", "キャラクターの表情と感情表現",
            "演出の斬新さ", "ページ全体の情報量", "台詞と絵の連携", "読者の引き込み",
            "アクションシーンの迫力", "間の使い方", "ページめくりの演出", "背景の効果的な使用"
        ],
        "defaults": ["コマ割りのリズムと視線誘導", "構図のダイナミズムと意図", "キャラクターの表情と感情表現"]
    },
    "完成原稿（画像 / PDF）": {
        "options": [
            "全体的な画力と魅力", "線の質と表現力（強弱・入り抜き）", "トーンワークと陰影表現",
            "背景の描き込みと世界観表現", "エフェクトや効果線の使い方", "キャラクターデザインの魅力",
            "商業誌レベルの完成度", "色彩センス（カラーの場合）", "文字・写植の美しさ", "印刷適性"
        ],
        "defaults": ["全体的な画力と魅力", "線の質と表現力（強弱・入り抜き）", "トーンワークと陰影表現"]
    }
}

EVALUATION_STYLES = {
    "厳格な編集者": "商業誌の基準で厳しく評価し、プロレベルを求める",
    "励ましの先輩": "良い点を多く見つけて励ましながら、建設的なアドバイスを提供",
    "技術指導者": "具体的な技術面での改善点を詳細に指摘し、上達方法を提案",
    "読者目線": "一般読者の視点から面白さや分かりやすさを重視して評価",
    "新人賞審査員": "新人賞の審査基準で将来性とポテンシャルを重視して評価"
}

# セッション状態の初期化
if 'projects' not in st.session_state: st.session_state.projects = []
if 'team_members' not in st.session_state: st.session_state.team_members = ["原作者", "作画担当", "アシスタント", "編集者"]
if 'idea_bank' not in st.session_state: st.session_state.idea_bank = []
if 'world_settings' not in st.session_state: st.session_state.world_settings = []
if 'characters' not in st.session_state: st.session_state.characters = []
if 'generated_content' not in st.session_state: st.session_state.generated_content = {}
if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = []
# APIクライアントの初期化
if 'openai_client' not in st.session_state: st.session_state.openai_client = None
if 'gemini_model' not in st.session_state: st.session_state.gemini_model = None

# --- API設定と呼び出し関数 (Gemini & OpenAI 対応) ---

def setup_apis():
    """サイドバーでの入力に基づきAPIクライアントをセットアップ"""
    # OpenAI
    openai_api_key = st.session_state.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        try:
            st.session_state.openai_client = OpenAI(api_key=openai_api_key)
        except Exception:
            st.session_state.openai_client = None # エラー時はNoneに戻す
    
    # Gemini
    google_api_key = st.session_state.get('google_api_key') or os.getenv('GOOGLE_API_KEY')
    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        except Exception:
            st.session_state.gemini_model = None # エラー時はNoneに戻す

def call_generative_ai(prompt_key, model, text_content="", image_data_list=None, **kwargs):
    """選択されたAIモデルを呼び出す汎用関数"""
    
    # kwargsにNoneが含まれているとformatでエラーになるため、空文字に変換
    for key, value in kwargs.items():
        if value is None:
            kwargs[key] = ""

    kwargs['text_content'] = text_content if text_content else "なし"
    prompt_text = GPTS_PROMPTS[prompt_key].format(**kwargs)
    
    try:
        # --- OpenAI (GPT)モデルの場合 ---
        if "gpt" in model.lower():
            if not st.session_state.openai_client:
                st.error("OpenAI APIキーが設定されていないか、無効です。サイドバーで設定してください。")
                return None

            user_messages = [{"type": "text", "text": prompt_text}]
            if image_data_list:
                for i, image_data in enumerate(image_data_list):
                    page_text = f"これは{i+1}ページ目の画像です。"
                    user_messages.append({"type": "text", "text": page_text})
                    user_messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    })
            
            messages = [
                {"role": "system", "content": "あなたは漫画制作のプロフェッショナルアシスタントです。"},
                {"role": "user", "content": user_messages}
            ]
            
            response = st.session_state.openai_client.chat.completions.create(
                model=model, messages=messages, temperature=0.7, max_tokens=4000
            )
            return response.choices[0].message.content

        # --- Google (Gemini)モデルの場合 ---
        elif "gemini" in model.lower():
            if not st.session_state.gemini_model:
                st.error("Google APIキーが設定されていないか、無効です。サイドバーで設定してください。")
                return None

            request_contents = [prompt_text]
            if image_data_list:
                for i, image_data in enumerate(image_data_list):
                    img_bytes = base64.b64decode(image_data)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    request_contents.append(f"これは{i+1}ページ目の画像です。")
                    request_contents.append(img)
            
            # 創作物の評価でブロックされにくくするための安全設定
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            response = st.session_state.gemini_model.generate_content(
                request_contents,
                safety_settings=safety_settings,
                generation_config={"temperature": 0.7}
            )
            return response.text
        
        else:
            st.error(f"サポートされていないモデルです: {model}")
            return None
            
    except Exception as e:
        st.error(f"AIモデル ({model}) の呼び出し中にエラーが発生しました: {e}")
        return None

# --- 補助関数 ---
def create_gantt_chart(tasks):
    if not tasks: return None
    df = pd.DataFrame(tasks)
    df['Start'] = pd.to_datetime(df['start_date'])
    df['Finish'] = pd.to_datetime(df['end_date'])
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="task_name", color="assignee", title="プロジェクトスケジュール", labels={"task_name": "タスク", "assignee": "担当者"})
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(height=max(400, len(tasks) * 30))
    return fig

def create_progress_chart(project):
    if 'tasks' not in project or not project['tasks']: return None
    tasks = project['tasks']
    completed = len([t for t in tasks if t.get('status') == '完了'])
    in_progress = len([t for t in tasks if t.get('status') == '進行中'])
    not_started = len([t for t in tasks if t.get('status') == '未着手'])
    fig = go.Figure(data=[
        go.Bar(name='完了', x=['進捗'], y=[completed], marker_color='#28a745'),
        go.Bar(name='進行中', x=['進捗'], y=[in_progress], marker_color='#ffc107'),
        go.Bar(name='未着手', x=['進捗'], y=[not_started], marker_color='#6c757d')
    ])
    fig.update_layout(barmode='stack', title='タスク進捗状況', yaxis_title='タスク数', height=300)
    return fig

# --- サイドバー (APIキー入力対応) ---
with st.sidebar:
    st.title("📚 漫画制作プロ管理")
    st.info("AI機能を使用するには、少なくとも1つのAPIキーを設定してください。")

    with st.expander("🔑 APIキー設定"):
        st.text_input("OpenAI APIキー", type="password", key='openai_api_key', on_change=setup_apis)
        st.text_input("Google APIキー (Gemini)", type="password", key='google_api_key', on_change=setup_apis)

    # on_changeでAPI設定が実行されるが、初回ロード時にも実行
    if 'first_load' not in st.session_state:
        setup_apis()
        st.session_state.first_load = False

    if st.session_state.openai_client:
        st.success("✅ OpenAI 接続済み")
    else:
        st.warning("❌ OpenAI 未接続")
    if st.session_state.gemini_model:
        st.success("✅ Google Gemini 接続済み")
    else:
        st.warning("❌ Google Gemini 未接続")
        
    st.divider()
    menu = st.radio("メニュー", ["🏠 ダッシュボード", "🚀 新規プロジェクト", "💡 アイデア工房", "📝 シナリオ作成", "👥 キャラクター工房", "🌍 世界観設定", "📅 スケジュール管理", "👥 チーム管理", "📊 分析・レポート", "✍️ アイデア・原稿評価"])

# --- メインコンテンツ ---

if menu == "🏠 ダッシュボード":
    st.title("📊 プロジェクトダッシュボード")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総プロジェクト", len(st.session_state.projects))
    with col2:
        active = len([p for p in st.session_state.projects if p.get('status') == '進行中'])
        st.metric("進行中", active)
    with col3:
        def get_upcoming_deadlines_count():
            count = 0
            today = datetime.now().date()
            one_week_later = today + timedelta(days=7)
            for project in st.session_state.projects:
                if 'tasks' in project:
                    for task in project['tasks']:
                        if task.get('status') != '完了':
                            try:
                                end_date = datetime.strptime(task['end_date'], "%Y-%m-%d").date()
                                if today <= end_date <= one_week_later:
                                    count += 1
                            except (ValueError, TypeError):
                                continue
            return count
        st.metric("今週の締切", get_upcoming_deadlines_count())
    with col4:
        st.metric("チームメンバー", len(st.session_state.team_members))
    st.subheader("📌 アクティブプロジェクト")
    active_projects = [p for p in st.session_state.projects if p.get('status') == '進行中']
    if active_projects:
        for project in active_projects:
            with st.expander(f"📖 {project['title']} - {project.get('genre', '未設定')}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**締切**: {project.get('deadline', '未設定')}")
                    st.write(f"**担当**: {project.get('assignee', '未定')}")
                    if 'tasks' in project and project['tasks']:
                        completed = len([t for t in project['tasks'] if t.get('status') == '完了'])
                        total = len(project['tasks'])
                        progress = completed / total if total > 0 else 0
                        st.progress(progress)
                        st.caption(f"進捗: {progress*100:.0f}% ({completed}/{total}タスク)")
                with col2:
                    if st.button(f"詳細を見る", key=f"view_{project['title']}"):
                        st.session_state.current_project_title = project['title']
                        st.success(f"「{project['title']}」を選択しました。「スケジュール管理」メニューに移動して詳細を確認してください。")
    else:
        st.info("進行中のプロジェクトはありません")
    st.subheader("📝 最近の活動")
    activities = [f"「{p['title']}」が作成されました - {p.get('created_at', '日時不明')}" for p in st.session_state.projects[-5:]]
    if activities:
        for activity in reversed(activities):
            st.write(f"• {activity}")
    else:
        st.write("まだ活動履歴がありません")

elif menu == "🚀 新規プロジェクト":
    st.title("🚀 新規プロジェクト作成")
    with st.form("new_project"):
        st.subheader("基本情報")
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("プロジェクト名", placeholder="例：次世代バトル漫画企画")
            project_type = st.selectbox("プロジェクトタイプ", ["連載準備", "読み切り", "コンペ用", "同人誌", "Web連載"])
            genre = st.selectbox("ジャンル", ["バトル", "ファンタジー", "SF", "恋愛", "ミステリー", "日常系", "ホラー", "スポーツ", "歴史"])
        with col2:
            deadline = st.date_input("締切日", min_value=datetime.today())
            assignee = st.selectbox("メイン担当者", st.session_state.team_members)
            priority = st.select_slider("優先度", options=["低", "中", "高", "緊急"], value="中")
        description = st.text_area("プロジェクト概要", height=100)
        st.subheader("初期タスク設定")
        use_template = st.checkbox("テンプレートを使用", value=True)
        if use_template and project_type in ["連載準備", "読み切り"]:
            st.info(f"{project_type}用のタスクテンプレートを使用します")
        submitted = st.form_submit_button("プロジェクトを作成", type="primary")
    if submitted and title:
        new_project = {
            "title": title, "type": project_type, "genre": genre, "deadline": deadline.strftime("%Y-%m-%d"),
            "assignee": assignee, "priority": priority, "description": description, "status": "進行中",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "tasks": []
        }
        if use_template and project_type in TASK_TEMPLATES:
            start_date = datetime.now()
            for task_template in TASK_TEMPLATES[project_type]:
                end_date = start_date + timedelta(days=task_template["duration"])
                task = {
                    "task_name": task_template["task"], "assignee": task_template["assignee"],
                    "start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d"),
                    "status": "未着手", "duration": task_template["duration"]
                }
                new_project["tasks"].append(task)
                start_date = end_date
        st.session_state.projects.append(new_project)
        st.success(f"プロジェクト「{title}」を作成しました！")
        st.balloons()

elif menu == "💡 アイデア工房":
    st.title("💡 アイデア工房 - MangaMaster AI")
    ai_model = st.selectbox("使用するAIモデル", ("gpt-4o", "gemini-1.5-pro-latest"), help="アイデア生成に使用するAIモデルを選択")
    tab1, tab2, tab3 = st.tabs(["クイック生成", "詳細生成", "アイデアバンク"])
    with tab1:
        st.subheader("クイックアイデア生成")
        col1, col2 = st.columns(2)
        with col1:
            quick_genre = st.selectbox("ジャンル", ["ファンタジー", "SF", "恋愛", "ミステリー", "アクション"], key="q_genre")
            quick_target = st.selectbox("ターゲット", ["少年向け", "少女向け", "青年向け", "女性向け"], key="q_target")
        with col2:
            quick_theme = st.text_input("テーマ/キーワード", placeholder="例：友情、成長、復讐", key="q_theme")
            quick_length = st.selectbox("想定規模", ["読み切り", "短期連載", "長期連載"], key="q_length")
        if st.button("💡 クイックアイデアを生成", type="primary"):
            with st.spinner(f"{ai_model}が考案中..."):
                response = call_generative_ai(
                    "manga_master", model=ai_model,
                    input_content=f"ジャンル: {quick_genre}, ターゲット: {quick_target}, テーマ: {quick_theme}, 想定規模: {quick_length}",
                    requirements="斬新で商業的にも成功可能な漫画のアイデアを3つ、タイトル、あらすじ、主要キャラクター、セールスポイントを明確にして提案してください。"
                )
                if response: st.session_state.generated_content['idea'] = response
        if 'idea' in st.session_state.generated_content:
            st.markdown("---")
            st.markdown(st.session_state.generated_content['idea'])
            if st.button("🏦 このアイデアをバンクに保存", key="save_quick_idea"):
                st.session_state.idea_bank.append({"title": f"クイックアイデア - {quick_genre}/{quick_theme}", "content": st.session_state.generated_content['idea'], "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.success("アイデアをバンクに保存しました！")
                del st.session_state.generated_content['idea']
    with tab2:
        st.subheader("詳細アイデア生成")
        with st.form("detailed_idea_form"):
            st.text("より詳細な要件を入力して、ユニークな物語を創造します。")
            col1, col2 = st.columns(2)
            with col1:
                genre = st.text_input("ジャンル・サブジャンル", "例: サイバーパンク・ノワール")
                setting = st.text_area("世界観・舞台設定", "例: 2242年のネオ・キョウト。企業が支配する巨大都市。")
            with col2:
                protagonist = st.text_area("主人公の設定", "例: 過去を失ったサイボーグ探偵。")
                antagonist = st.text_area("敵役・障害", "例: 主人公を改造した巨大複合企業。")
            plot_twist = st.text_input("物語に入れたい意外な展開", "例: ヒロインが実は敵のスパイだった。")
            submitted = st.form_submit_button("🌟 詳細アイデアを生成", type="primary")
            if submitted:
                with st.spinner(f"{ai_model}が物語を構築中..."):
                    response = call_generative_ai(
                        "manga_master", model=ai_model,
                        input_content=f"ジャンル: {genre}, 世界観: {setting}, 主人公: {protagonist}, 敵役: {antagonist}, 必須要素: {plot_twist}",
                        requirements="上記の設定を元に、魅力的な連載漫画の第1話のあらすじと、今後の展開の可能性について詳細に提案してください。"
                    )
                    if response: st.session_state.generated_content['detailed_idea'] = response
        if 'detailed_idea' in st.session_state.generated_content:
            st.markdown("---")
            st.markdown(st.session_state.generated_content['detailed_idea'])
            if st.button("🏦 この詳細アイデアをバンクに保存", key="save_detailed_idea"):
                 st.session_state.idea_bank.append({"title": f"詳細アイデア - {genre}", "content": st.session_state.generated_content['detailed_idea'], "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
                 st.success("アイデアをバンクに保存しました！")
                 del st.session_state.generated_content['detailed_idea']
    with tab3:
        st.subheader("アイデアバンク")
        if not st.session_state.idea_bank:
            st.info("まだ保存されたアイデアはありません。")
        else:
            for i, item in enumerate(st.session_state.idea_bank):
                with st.expander(f"💡 {item['title']} ({item['created_at']})"):
                    st.markdown(item['content'])
                    if st.button("削除", key=f"del_idea_{i}", type="secondary"):
                        st.session_state.idea_bank.pop(i)
                        st.rerun()

elif menu == "📝 シナリオ作成":
    st.title("📝 シナリオ作成工房")
    ai_model = st.selectbox("使用するAIモデル", ("gpt-4o", "gemini-1.5-pro-latest"))
    with st.form("scenario_form"):
        st.info("物語の骨子となる情報を入力し、具体的なシーンのシナリオを生成します。")
        scenario_base = st.text_area("シナリオのベースとなるプロットや状況", height=150, placeholder="例：主人公のアキラが、長年追い求めていた伝説の剣をついに発見するシーン。しかし、そこにはライバルのカイトも現れる。")
        scene_details = st.text_area("シーンの詳細や演出の要望", height=100, placeholder="例：洞窟の奥深く、剣は台座に突き刺さり青白い光を放っている。アキラとカイトの緊張感のある対峙を強調してほしい。")
        submitted = st.form_submit_button("📜 シナリオを生成", type="primary")
        if submitted:
            with st.spinner(f"{ai_model}が執筆中..."):
                response = call_generative_ai(
                    "scenario_writer", model=ai_model,
                    scenario_base=scenario_base, scene_details=scene_details
                )
                if response:
                    st.session_state.generated_content['scenario'] = response
    if 'scenario' in st.session_state.generated_content:
        st.markdown("---")
        st.subheader("生成されたシナリオ")
        st.markdown(st.session_state.generated_content['scenario'])

elif menu == "👥 キャラクター工房":
    st.title("👥 キャラクター工房")
    ai_model = st.selectbox("使用するAIモデル", ("gpt-4o", "gemini-1.5-pro-latest"), key="char_model")
    tab1, tab2, tab3 = st.tabs(["キャラクター作成", "キャラクター一覧", "キャラクターアーク設計"])
    with tab1:
        st.subheader("🎨 新規キャラクター作成")
        with st.form("character_form"):
            char_name = st.text_input("キャラクター名")
            col1, col2, col3 = st.columns(3)
            with col1:
                char_age = st.number_input("年齢", 0, 120, 16)
            with col2:
                char_gender = st.selectbox("性別", ["男性", "女性", "その他", "不明"])
            with col3:
                char_role = st.selectbox("役割", ["主人公", "ヒロイン", "相棒", "ライバル", "師匠", "敵役", "その他"])
            personality = st.text_area("性格・内面（キーワードでOK）", placeholder="例：クール、負けず嫌い、実は寂しがり屋、猫が好き")
            backstory = st.text_area("バックストーリー（キーワードでOK）", placeholder="例：孤児院育ち、謎の組織に追われている、失われた記憶")
            abilities = st.text_area("能力・スキル", placeholder="例：炎を操る能力、天才的なハッキング技術")
            submitted = st.form_submit_button("🎨 キャラクターを生成", type="primary")
            if submitted and char_name:
                with st.spinner(f"{ai_model}がキャラクターを構築中..."):
                    response = call_generative_ai(
                        "character_developer", model=ai_model,
                        character_info=f"名前: {char_name}, 年齢: {char_age}, 性別: {char_gender}, 役割: {char_role}, 性格: {personality}, 背景: {backstory}, 能力: {abilities}"
                    )
                    if response:
                        new_char = {"name": char_name, "details": response, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
                        st.session_state.characters.append(new_char)
                        st.success(f"キャラクター「{char_name}」が作成され、一覧に追加されました！")
                        st.balloons()
    with tab2:
        st.subheader("📜 キャラクター一覧")
        if not st.session_state.characters:
            st.info("まだ作成されたキャラクターがいません。")
        else:
            for i, char in enumerate(st.session_state.characters):
                with st.expander(f"👤 {char['name']}"):
                    st.markdown(char['details'])
                    if st.button("削除", key=f"del_char_{i}", type="secondary"):
                        st.session_state.characters.pop(i)
                        st.rerun()
    with tab3:
        # この機能はAIを使用しないため、モデル選択は不要
        st.subheader("📈 キャラクターアーク設計")
        if not st.session_state.characters:
            st.warning("先に「キャラクター作成」タブでキャラクターを作成してください。")
        else:
            char_names = [c['name'] for c in st.session_state.characters]
            arc_character = st.selectbox("対象キャラクター", char_names)
            start_point = st.text_area("開始時点の状態", placeholder="物語開始時のキャラクターの状態、価値観、欠点など")
            end_point = st.text_area("到達点", placeholder="物語終了時に到達すべき状態、成長した姿など")
            key_events = st.text_area("成長のきっかけとなる重要イベント（箇条書き）", placeholder="例：\n- 師匠との出会い\n- ライバルへの敗北\n- 守るべきものができる")
            if st.button("📈 成長アークを生成"):
                with st.spinner("成長の軌跡を設計中..."):
                    st.success(f"{arc_character}のキャラクターアークが設計されました！")
                    st.markdown(f"#### {arc_character}の成長物語")
                    st.write(f"**【序盤】** {start_point}")
                    st.write("**【転機】**")
                    st.code(key_events, language='markdown')
                    st.write(f"**【終盤】** {end_point}")
                    stages = ["序盤", "転機", "クライマックス", "終盤"]
                    growth_values = [20, 50, 80, 100]
                    fig = go.Figure(go.Scatter(x=stages, y=growth_values, mode='lines+markers', name='成長曲線', line=dict(color='royalblue', width=3), marker=dict(size=10)))
                    fig.update_layout(title=f"{arc_character}の成長アーク", xaxis_title="物語の進行", yaxis_title="成長度", height=400)
                    st.plotly_chart(fig, use_container_width=True)

elif menu == "🌍 世界観設定":
    st.title("🌍 世界観設定工房")
    ai_model = st.selectbox("使用するAIモデル", ("gpt-4o", "gemini-1.5-pro-latest"), key="world_model")
    tab1, tab2, tab3 = st.tabs(["世界観生成", "設定集", "地図作成支援"])
    with tab1:
        st.subheader("世界観の基本設定")
        with st.form("world_building"):
            world_name = st.text_input("世界/舞台の名前")
            world_type = st.selectbox("世界タイプ", ["現実世界ベース", "完全架空世界", "パラレルワールド", "未来世界", "過去世界", "異次元"])
            col1, col2 = st.columns(2)
            with col1:
                geography = st.text_area("地理・環境", placeholder="大陸の配置、気候、特殊な地形など")
                technology = st.text_area("技術/魔法体系", placeholder="利用可能な技術、魔法の仕組み、制限など")
            with col2:
                society = st.text_area("社会・文化", placeholder="政治体制、経済、階級制度など")
                history = st.text_area("歴史・伝承", placeholder="建国神話、大事件、現在への影響など")
            special_rules = st.text_area("この世界特有のルール", placeholder="物理法則の違い、特殊な制約、独自の概念など")
            submitted = st.form_submit_button("🌍 世界観を構築", type="primary")
        if submitted and world_name:
            with st.spinner(f"{ai_model}が世界を創造中..."):
                response = call_generative_ai(
                    "world_builder", model=ai_model,
                    world_base=f"世界名: {world_name}, タイプ: {world_type}, 地理: {geography}, 技術: {technology}, 社会: {society}, 歴史: {history}, 特殊ルール: {special_rules}",
                    additional_requests="矛盾のない、魅力的で独創的な世界観を構築してください。読者がワクワクするような設定を盛り込んでください。"
                )
                if response:
                    st.session_state.generated_content['world'] = {"name": world_name, "content": response}
        if 'world' in st.session_state.generated_content:
            st.markdown("---")
            st.subheader(f"生成された世界観: {st.session_state.generated_content['world']['name']}")
            st.markdown(st.session_state.generated_content['world']['content'])
            if st.button("💾 この設定を保存", key="save_world"):
                world_data = st.session_state.generated_content['world']
                st.session_state.world_settings.append({"name": world_data['name'], "content": world_data['content'], "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.success("世界観設定を保存しました！")
                del st.session_state.generated_content['world']
    with tab2:
        st.subheader("📚 設定集")
        if not st.session_state.world_settings:
            st.info("まだ保存された世界観設定はありません。")
        else:
            for i, setting in enumerate(st.session_state.world_settings):
                with st.expander(f"🌍 {setting['name']} ({setting['created_at']})"):
                    st.markdown(setting['content'])
                    if st.button("削除", key=f"del_world_{i}", type="secondary"):
                        st.session_state.world_settings.pop(i)
                        st.rerun()
    with tab3:
        st.subheader("🗺️ 地図作成支援")
        st.info("地図に含めたい要素を文章で説明すると、AIが具体的な描写や配置のアイデアを提供します。")
        map_description = st.text_area("地図の説明", height=150, placeholder="例：中央に巨大なクレーター湖があり、その周りを険しい山脈が囲んでいる。北の森にはエルフの隠れ里が、南の平原には人間の王国が広がっている。")
        if st.button("🗺️ 地図作成ガイドを生成"):
            with st.spinner(f"{ai_model}が地図のアイデアを考案中..."):
                response = call_generative_ai(
                    "world_builder", model=ai_model,
                    world_base=f"地図のアイデア: {map_description}",
                    additional_requests="この説明に基づき、より詳細な地理的特徴、都市や村の具体的な位置、街道、ダンジョンなどの興味深い場所のアイデアを箇条書きで提案してください。"
                )
                if response:
                    st.markdown(response)
            st.warning("※実際の地図は画像編集ソフトで作成してください。")

elif menu == "📅 スケジュール管理":
    st.title("📅 スケジュール管理")
    if st.session_state.projects:
        project_titles = [p['title'] for p in st.session_state.projects]
        default_index = 0
        if 'current_project_title' in st.session_state and st.session_state.current_project_title in project_titles:
            default_index = project_titles.index(st.session_state.current_project_title)
        selected_project_title = st.selectbox("プロジェクトを選択", project_titles, index=default_index)
        st.session_state.current_project_title = selected_project_title
        project_index = next((i for i, p in enumerate(st.session_state.projects) if p['title'] == selected_project_title), None)
        if project_index is not None:
            project = st.session_state.projects[project_index]
            tab1, tab2, tab3 = st.tabs(["ガントチャート", "タスク管理", "締切アラート"])
            with tab1:
                st.subheader("📊 プロジェクトタイムライン")
                if 'tasks' in project and project['tasks']:
                    gantt = create_gantt_chart(project['tasks'])
                    progress_chart = create_progress_chart(project)
                    if gantt:
                        st.plotly_chart(gantt, use_container_width=True)
                    if progress_chart:
                        st.plotly_chart(progress_chart, use_container_width=True)
                else:
                    st.info("このプロジェクトにはタスクがありません。「タスク管理」タブで追加してください。")
            with tab2:
                st.subheader("✅ タスク管理")
                with st.expander("➕ 新規タスク追加"):
                    with st.form("add_task_form"):
                        col1, col2 = st.columns(2)
                        with col1:
                            task_name = st.text_input("タスク名")
                            assignee = st.selectbox("担当者", st.session_state.team_members, key="task_assignee")
                        with col2:
                            start_date = st.date_input("開始日", value=datetime.today())
                            duration = st.number_input("期間（日）", min_value=1, value=3)
                        if st.form_submit_button("タスクを追加"):
                            if task_name:
                                new_task = {
                                    "task_name": task_name, "assignee": assignee,
                                    "start_date": start_date.strftime("%Y-%m-%d"),
                                    "end_date": (start_date + timedelta(days=duration)).strftime("%Y-%m-%d"),
                                    "status": "未着手"
                                }
                                project['tasks'].append(new_task)
                                st.success(f"タスク「{task_name}」を追加しました！")
                                st.rerun()
                if 'tasks' in project and project['tasks']:
                    st.markdown("### 📋 タスク一覧")
                    for i, task in enumerate(project['tasks']):
                        cols = st.columns([4, 2, 2, 1])
                        cols[0].write(f"**{task['task_name']}** ({task['assignee']})")
                        cols[1].write(f"🗓️ {task['start_date']} ~ {task['end_date']}")
                        status_options = ["未着手", "進行中", "完了", "保留"]
                        current_status_index = status_options.index(task.get('status', '未着手'))
                        new_status = cols[2].selectbox("ステータス", status_options, index=current_status_index, key=f"status_{project_index}_{i}", label_visibility="collapsed")
                        if new_status != task['status']:
                            task['status'] = new_status
                            st.rerun()
                        if cols[3].button("🗑️", key=f"del_task_{project_index}_{i}", help="タスクを削除"):
                            project['tasks'].pop(i)
                            st.rerun()
            with tab3:
                st.subheader("⏰ 締切アラート")
                today = datetime.now().date()
                urgent_tasks, upcoming_tasks = [], []
                if 'tasks' in project:
                    for task in project['tasks']:
                        if task.get('status') != '完了':
                            end_date = datetime.strptime(task['end_date'], "%Y-%m-%d").date()
                            days_left = (end_date - today).days
                            if days_left < 0:
                                urgent_tasks.append((task, days_left))
                            elif days_left <= 7:
                                upcoming_tasks.append((task, days_left))
                if urgent_tasks:
                    st.error("🚨 期限超過タスク")
                    for task, days in urgent_tasks:
                        st.write(f"• **{task['task_name']}** - {abs(days)}日超過 ({task['assignee']})")
                if upcoming_tasks:
                    st.warning("⚠️ 締切間近タスク (7日以内)")
                    for task, days in upcoming_tasks:
                        st.write(f"• **{task['task_name']}** - 残り{days}日 ({task['assignee']})")
                if not urgent_tasks and not upcoming_tasks:
                    st.success("✅ 締切間近のタスクはありません。")
    else:
        st.info("まずは「新規プロジェクト」メニューからプロジェクトを作成してください。")

elif menu == "👥 チーム管理":
    st.title("👥 チーム管理")
    tab1, tab2, tab3 = st.tabs(["メンバー一覧", "役割分担", "作業負荷"])
    with tab1:
        st.subheader("チームメンバー")
        with st.form("add_member"):
            new_member = st.text_input("新しいメンバー名")
            if st.form_submit_button("メンバーを追加"):
                if new_member and new_member not in st.session_state.team_members:
                    st.session_state.team_members.append(new_member)
                    st.success(f"{new_member}をチームに追加しました！")
                    st.rerun()
        st.markdown("### 現在のチーム")
        for member in st.session_state.team_members:
            col1, col2 = st.columns([4, 1])
            col1.write(f"👤 **{member}**")
            if col2.button("削除", key=f"del_member_{member}", type="secondary"):
                st.session_state.team_members.remove(member)
                st.rerun()
    with tab2:
        st.subheader("役割分担マトリックス (RACIチャート例)")
        roles_data = {
            "タスク": ["ストーリー", "キャラデザ", "ネーム", "作画", "仕上げ", "進行管理"],
            "原作者": ["実行責任者(R)", "承認者(A)", "実行責任者(R)", "協業(C)", "", "情報提供(I)"],
            "作画担当": ["協業(C)", "実行責任者(R)", "実行責任者(R)", "実行責任者(R)", "協業(C)", "協業(C)"],
            "アシスタント": ["", "", "", "協業(C)", "実行責任者(R)", ""],
            "編集者": ["承認者(A)", "承認者(A)", "承認者(A)", "情報提供(I)", "", "実行責任者(R)"]
        }
        df_roles = pd.DataFrame(roles_data).set_index("タスク")
        st.dataframe(df_roles, use_container_width=True)
        st.caption("R: Responsible (実行責任者), A: Accountable (承認者), C: Consulted (協業), I: Informed (情報提供)")
    with tab3:
        st.subheader("作業負荷分析")
        all_tasks = [task for proj in st.session_state.projects for task in proj.get('tasks', [])]
        if all_tasks:
            df_tasks = pd.DataFrame(all_tasks)
            assignee_counts = df_tasks.groupby(['assignee', 'status']).size().unstack(fill_value=0)
            fig = go.Figure()
            statuses = ['完了', '進行中', '未着手', '保留']
            colors = {'完了': '#28a745', '進行中': '#ffc107', '未着手': '#6c757d', '保留': '#17a2b8'}
            for status in statuses:
                if status in assignee_counts.columns:
                    fig.add_trace(go.Bar(name=status, x=assignee_counts.index, y=assignee_counts[status], marker_color=colors.get(status)))
            fig.update_layout(barmode='stack', title='メンバー別タスク負荷', xaxis_title='メンバー', yaxis_title='タスク数')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("タスクデータがありません。")

elif menu == "📊 分析・レポート":
    st.title("📊 分析・レポート")
    tab1, tab2, tab3 = st.tabs(["プロジェクト分析", "進捗レポート", "データエクスポート"])
    with tab1:
        st.subheader("プロジェクト分析")
        if st.session_state.projects:
            df_projects = pd.DataFrame(st.session_state.projects)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### プロジェクトステータス")
                status_counts = df_projects['status'].value_counts()
                st.dataframe(status_counts)
            with col2:
                st.markdown("#### ジャンル分布")
                genre_counts = df_projects['genre'].value_counts()
                fig_pie = px.pie(values=genre_counts.values, names=genre_counts.index, title="ジャンル分布")
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("分析するプロジェクトがありません。")
    with tab2:
        st.subheader("進捗レポート生成")
        st.info("この機能は現在開発中です。")
    with tab3:
        st.subheader("データエクスポート")
        export_data_options = ["プロジェクト", "タスク一覧", "キャラクター", "世界観設定", "アイデアバンク", "評価履歴"]
        selected_data = st.multiselect("エクスポートするデータを選択", export_data_options, default=["プロジェクト"])
        if st.button("📤 エクスポートデータを準備", type="primary"):
            export_content = {"exported_at": datetime.now().isoformat()}
            if "プロジェクト" in selected_data: export_content["projects"] = st.session_state.projects
            if "タスク一覧" in selected_data: export_content["all_tasks"] = [task for proj in st.session_state.projects for task in proj.get('tasks', [])]
            if "キャラクター" in selected_data: export_content["characters"] = st.session_state.characters
            if "世界観設定" in selected_data: export_content["world_settings"] = st.session_state.world_settings
            if "アイデアバンク" in selected_data: export_content["idea_bank"] = st.session_state.idea_bank
            if "評価履歴" in selected_data: export_content["evaluation_results"] = st.session_state.evaluation_results
            json_str = json.dumps(export_content, ensure_ascii=False, indent=2)
            st.download_button(label="📥 JSON形式でダウンロード", data=json_str, file_name=f"manga_pro_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

elif menu == "✍️ アイデア・原稿評価":
    st.title("✍️ 強化版アイデア・原稿評価システム")
    st.info("📝 テキスト、🖼️ 画像、📄 PDFファイルをアップロードして、プロ編集者レベルのAI評価を受けましょう。")
    
    ai_model = st.selectbox("使用するAIモデル", ("gpt-4o", "gemini-1.5-pro-latest"), key="eval_model")

    evaluation_mode = st.radio(
        "評価モードを選択",
        ["📋 全体評価", "📖 ページ別詳細評価", "📊 評価履歴"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if evaluation_mode == "📋 全体評価":
        st.header("📋 全体評価")
        with st.expander("⚙️ 評価設定", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                eval_type = st.selectbox("評価対象の種類", ["プロット / テキスト", "ネーム（画像 / PDF）", "完成原稿（画像 / PDF）"])
                evaluation_style_key = st.selectbox("評価者のスタイル", list(EVALUATION_STYLES.keys()), help="評価者の視点とトーンを選択してください")
                evaluation_style = EVALUATION_STYLES[evaluation_style_key]
                detail_level = st.select_slider("評価の詳細度", options=["簡潔", "標準", "詳細", "徹底"], value="標準")
            with col2:
                current_eval_points = EVALUATION_OPTIONS[eval_type]
                selected_eval_points = st.multiselect("評価の観点（複数選択可）", current_eval_points["options"], default=current_eval_points["defaults"])
                special_instructions = st.text_area("特別な指示・注目点", placeholder="例：初心者向けのアドバイス重視、商業性を特に重視、など", height=100)

        file_types = {"プロット / テキスト": ["txt", "md"], "ネーム（画像 / PDF）": ["png", "jpg", "jpeg", "pdf"], "完成原稿（画像 / PDF）": ["png", "jpg", "jpeg", "pdf"]}
        uploaded_files = st.file_uploader(f"📁 評価したい「{eval_type}」ファイルをアップロード（複数可）", type=file_types[eval_type], accept_multiple_files=True)

        if uploaded_files:
            text_content, image_data_list = "", []
            st.markdown("---")
            st.subheader("📖 アップロードされた内容のプレビュー")
            with st.spinner("ファイルを処理中..."):
                for uploaded_file in uploaded_files:
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    if file_ext in [".txt", ".md"]:
                        content = uploaded_file.getvalue().decode("utf-8")
                        text_content += f"\n\n--- ファイル: {uploaded_file.name} ---\n{content}"
                    elif file_ext == ".pdf":
                        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                        for page in pdf_doc:
                            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                            img_bytes = pix.tobytes("png")
                            image_data_list.append(base64.b64encode(img_bytes).decode('utf-8'))
                    elif file_ext in [".png", ".jpg", ".jpeg"]:
                        img_bytes = uploaded_file.getvalue()
                        image_data_list.append(base64.b64encode(img_bytes).decode('utf-8'))

            if text_content:
                with st.expander("📝 テキスト内容を表示"):
                    st.text_area("読み込まれたテキスト", text_content, height=200, disabled=True)
            if image_data_list:
                st.write(f"🖼️ **画像プレビュー** ({len(image_data_list)}ページ)")
                cols = st.columns(min(6, len(image_data_list)))
                for i, img_data in enumerate(image_data_list):
                    cols[i % 6].image(f"data:image/png;base64,{img_data}", caption=f"P.{i+1}", width=120)

            if st.button(f"🤖 AI({ai_model})による「{eval_type}」の全体評価を開始", type="primary", use_container_width=True):
                with st.spinner(f"🔍 AI編集者が総合的に評価中..."):
                    response = call_generative_ai(
                        "manuscript_evaluator", model=ai_model, text_content=text_content, image_data_list=image_data_list,
                        content_type=eval_type, evaluation_points=", ".join(selected_eval_points),
                        detail_level=detail_level, evaluation_style=evaluation_style,
                        special_instructions=special_instructions, page_count=len(image_data_list),
                        evaluation_format="評価は総合評価、良い点、改善点、具体的な提案、総括の5つの項目で構成してください。",
                        page_specific_format=""
                    )
                    if response:
                        st.markdown("---")
                        st.subheader("📊 AI編集者からの総合評価")
                        st.markdown(response)
                        result = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "type": "全体評価",
                            "model": ai_model, "content_type": eval_type, "evaluation_style": evaluation_style_key,
                            "detail_level": detail_level, "evaluation_points": selected_eval_points,
                            "result": response, "text_content": text_content, "image_data": image_data_list
                        }
                        st.session_state.evaluation_results.append(result)
                        st.success("✅ 評価完了！評価履歴に保存されました。")

    elif evaluation_mode == "📖 ページ別詳細評価":
        st.header("📖 ページ別詳細評価")
        with st.expander("⚙️ ページ別評価設定", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                page_eval_type = st.selectbox("評価対象", ["ネーム（画像 / PDF）", "完成原稿（画像 / PDF）"], key="page_eval_type")
                page_eval_points = st.multiselect("ページ評価の観点", EVALUATION_OPTIONS[page_eval_type]["options"], default=EVALUATION_OPTIONS[page_eval_type]["defaults"], key="page_eval_points")
            with col2:
                focus_areas = st.text_area("特に注目したい要素", placeholder="例：アクションシーンの迫力、キャラクターの表情など", height=100, key="page_focus_areas")
                eval_all_pages = st.checkbox("全ページを一括評価", value=True, key="eval_all")
                eval_page_range = ""
                if not eval_all_pages:
                    eval_page_range = st.text_input("評価ページ指定", placeholder="例: 1,3,5-7", help="カンマ区切り、ハイフンで範囲指定", key="page_range")

        uploaded_files_page = st.file_uploader("📁 ページ別評価用のファイル（画像/PDF）をアップロード", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

        if uploaded_files_page:
            image_data_list_page, page_info_list = [], []
            with st.spinner("ファイルを処理中..."):
                for uploaded_file in uploaded_files_page:
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    if file_ext == ".pdf":
                        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                        for i, page in enumerate(pdf_doc):
                            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                            img_bytes = pix.tobytes("png")
                            image_data_list_page.append(base64.b64encode(img_bytes).decode('utf-8'))
                            page_info_list.append(f"{uploaded_file.name} - P.{i+1}")
                    elif file_ext in [".png", ".jpg", ".jpeg"]:
                        img_bytes = uploaded_file.getvalue()
                        image_data_list_page.append(base64.b64encode(img_bytes).decode('utf-8'))
                        page_info_list.append(uploaded_file.name)
            
            st.info(f"✅ {len(image_data_list_page)}ページの読み込みが完了しました。")
            
            pages_to_evaluate_indices = []
            if eval_all_pages:
                pages_to_evaluate_indices = list(range(len(image_data_list_page)))
            elif eval_page_range:
                try:
                    for part in eval_page_range.split(','):
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            pages_to_evaluate_indices.extend(range(start - 1, min(end, len(image_data_list_page))))
                        else:
                            pages_to_evaluate_indices.append(int(part) - 1)
                    pages_to_evaluate_indices = sorted(list(set(p for p in pages_to_evaluate_indices if 0 <= p < len(image_data_list_page))))
                except ValueError:
                    st.error("ページ指定の形式が正しくありません。")

            if pages_to_evaluate_indices:
                st.write(f"**評価対象**: {len(pages_to_evaluate_indices)}ページ ({', '.join(map(lambda x: str(x+1), pages_to_evaluate_indices))})")
                if st.button(f"🔍 {len(pages_to_evaluate_indices)}ページの個別評価を({ai_model})で開始", type="primary", use_container_width=True):
                    st.markdown("---")
                    st.subheader("📖 ページ別詳細評価結果")
                    page_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, page_idx in enumerate(pages_to_evaluate_indices):
                        status_text.text(f"🔍 {page_idx + 1}ページ目を評価中... ({i + 1}/{len(pages_to_evaluate_indices)})")
                        result = call_generative_ai(
                            "page_evaluator", model=ai_model,
                            image_data_list=[image_data_list_page[page_idx]],
                            page_number=page_idx + 1,
                            evaluation_points=", ".join(page_eval_points),
                            focus_areas=focus_areas if focus_areas else "特になし"
                        )
                        if result:
                            page_results.append({"page_number": page_idx + 1, "page_info": page_info_list[page_idx], "result": result})
                            with st.expander(f"📄 **{page_idx + 1}ページ目** の評価結果", expanded=True):
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(f"data:image/png;base64,{image_data_list_page[page_idx]}", caption=page_info_list[page_idx])
                                with col2:
                                    st.markdown(result)
                        progress_bar.progress((i + 1) / len(pages_to_evaluate_indices))
                    
                    status_text.success("✅ 全ページの評価が完了しました！")
                    if page_results:
                        full_result = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "type": "ページ別評価",
                            "model": ai_model, "content_type": page_eval_type,
                            "evaluation_points": page_eval_points, "focus_areas": focus_areas,
                            "page_results": page_results, "image_data": image_data_list_page,
                            "evaluated_indices": pages_to_evaluate_indices
                        }
                        st.session_state.evaluation_results.append(full_result)
                        st.success("評価結果を履歴に保存しました。")
            else:
                st.warning("評価対象のページがありません。設定を確認してください。")

    elif evaluation_mode == "📊 評価履歴":
        st.header("📊 評価履歴")
        if not st.session_state.evaluation_results:
            st.info("まだ評価履歴がありません。「全体評価」または「ページ別詳細評価」を実行してください。")
        else:
            filtered_results = sorted(st.session_state.evaluation_results, key=lambda x: x["timestamp"], reverse=True)
            
            st.write(f"総評価数: {len(filtered_results)}件")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ 全履歴を削除", type="secondary"):
                    st.session_state.evaluation_results = []
                    st.rerun()
            with col2:
                export_json_all = json.dumps(filtered_results, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📤 全履歴をJSONでエクスポート",
                    data=export_json_all,
                    file_name=f"manga_pro_evaluation_history_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.divider()

            for i, result in enumerate(filtered_results):
                icon = "📋" if result["type"] == "全体評価" else "📖"
                with st.expander(f"{icon} {result['type']} - {result['timestamp']} (by {result.get('model', 'N/A')})"):
                    
                    tab1, tab2 = st.tabs(["📝 評価結果", "🖼️ 評価対象コンテンツ"])
                    
                    with tab1:
                        if result["type"] == "全体評価":
                            st.markdown(f"**評価スタイル**: {result['evaluation_style']} | **詳細度**: {result['detail_level']}")
                            st.markdown(f"**評価観点**: {', '.join(result['evaluation_points'])}")
                            st.markdown("---")
                            st.markdown(result['result'])
                        elif result["type"] == "ページ別評価":
                            st.markdown(f"**評価ページ数**: {len(result['page_results'])} / {len(result['image_data'])}")
                            st.markdown(f"**評価観点**: {', '.join(result['evaluation_points'])}")
                            if result.get('focus_areas'): st.markdown(f"**注目要素**: {result['focus_areas']}")
                            st.markdown("---")
                            for page_res in result['page_results']:
                                with st.container():
                                    st.subheader(f"📄 {page_res['page_number']}ページ目")
                                    st.markdown(page_res['result'])
                                    st.divider()
                    
                    with tab2:
                        st.markdown("**評価時に使用されたコンテンツ**")
                        if result.get("text_content"):
                            st.text_area("テキストコンテンツ", result["text_content"], height=150, disabled=True, key=f"history_text_{i}")
                        
                        if result.get("image_data"):
                            image_list = result["image_data"]
                            evaluated_indices = result.get("evaluated_indices", []) if result['type'] == 'ページ別評価' else list(range(len(image_list)))
                            
                            st.write(f"画像コンテンツ ({len(image_list)}ページ)")
                            cols = st.columns(min(6, len(image_list)))
                            for j, img_data in enumerate(image_list):
                                caption = f"P.{j+1}"
                                use_border = j in evaluated_indices
                                
                                # 評価対象ページに枠線をつける
                                if use_border:
                                    cols[j % 6].markdown(f'<div style="border: 2px solid #ff4b4b; padding: 2px; border-radius: 5px; text-align: center;">', unsafe_allow_html=True)
                                    cols[j % 6].image(f"data:image/png;base64,{img_data}", width=100)
                                    cols[j % 6].caption(caption)
                                    cols[j % 6].markdown('</div>', unsafe_allow_html=True)
                                else:
                                    with cols[j % 6]:
                                        st.image(f"data:image/png;base64,{img_data}", width=100)
                                        st.caption(caption)

                    st.divider()
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        result_json = json.dumps(result, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="📄 この評価をダウンロード",
                            data=result_json,
                            file_name=f"evaluation_{result['type'].replace(' ', '_')}_{result['timestamp'].replace(':', '-').replace(' ', '_')}.json",
                            mime="application/json",
                            key=f"download_hist_{i}"
                        )
                    with d_col2:
                        if st.button("🗑️ この評価を削除", key=f"del_hist_{i}", type="secondary"):
                            # 削除対象を特定するためにインデックスではなく、タイムスタンプで検索
                            original_index = -1
                            for idx, item in enumerate(st.session_state.evaluation_results):
                                if item['timestamp'] == result['timestamp']:
                                    original_index = idx
                                    break
                            if original_index != -1:
                                st.session_state.evaluation_results.pop(original_index)
                            st.rerun()

# フッター
st.divider()
st.caption("🤖 Powered by OpenAI, Google Gemini & Streamlit | 漫画制作プロフェッショナル管理システム v3.0 (Dual AI)")

# --- END OF COMPLETE FILE ---