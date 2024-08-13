from typing import List, Dict
import httpx
import gradio as gr
import pandas as pd
from huggingface_hub import HfApi, ModelCard

def search_hub(query: str, search_type: str) -> pd.DataFrame:
    api = HfApi()
    if search_type == "Models":
        results = api.list_models(search=query)
        data = [{"id": model.modelId, "author": model.author, "downloads": model.downloads, "link": f"https://huggingface.co/{model.modelId}"} for model in results]
    elif search_type == "Datasets":
        results = api.list_datasets(search=query)
        data = [{"id": dataset.id, "author": dataset.author, "downloads": dataset.downloads, "link": f"https://huggingface.co/datasets/{dataset.id}"} for dataset in results]
    elif search_type == "Spaces":
        results = api.list_spaces(search=query)
        data = [{"id": space.id, "author": space.author, "link": f"https://huggingface.co/spaces/{space.id}"} for space in results]
    else:
        data = []
    
    # Add numbering and format the link
    for i, item in enumerate(data, 1):
        item['number'] = i
        item['formatted_link'] = format_link(item, i, search_type)
    
    return pd.DataFrame(data)

def format_link(item: Dict, number: int, search_type: str) -> str:
    link = item['link']
    readme_link = f"{link}/blob/main/README.md"
    title = f"{number}. {item['id']}"
    
    metadata = f"Author: {item['author']}"
    if 'downloads' in item:
        metadata += f", Downloads: {item['downloads']}"
    
    html = f"""
    <div style="margin-bottom: 10px;">
        <strong>{title}</strong><br>
        <a href="{link}" target="_blank" style="color: #4a90e2; text-decoration: none;">View {search_type[:-1]}</a> | 
        <a href="{readme_link}" target="_blank" style="color: #4a90e2; text-decoration: none;">View README</a><br>
        <small>{metadata}</small>
    </div>
    """
    return html

def display_results(df: pd.DataFrame):
    if df is not None and not df.empty:
        html = "<div style='max-height: 400px; overflow-y: auto;'>"
        for _, row in df.iterrows():
            html += row['formatted_link']
        html += "</div>"
        return html
    else:
        return "<p>No results found.</p>"

def load_metadata(evt: gr.SelectData, df: pd.DataFrame, search_type: str):
    if df is not None and not df.empty and evt.index[0] < len(df):
        item_id = df.iloc[evt.index[0]]['id']
        
        if search_type == "Models":
            try:
                card = ModelCard.load(item_id)
                return str(card)
            except Exception as e:
                return f"Error loading model card: {str(e)}"
        elif search_type == "Datasets":
            api = HfApi()
            metadata = api.dataset_info(item_id)
            return str(metadata)
        elif search_type == "Spaces":
            api = HfApi()
            metadata = api.space_info(item_id)
            return str(metadata)
        else:
            return ""
    else:
        return ""

def SwarmyTime(data: List[Dict]) -> Dict:
    """
    Aggregates all content from the given data.
    
    :param data: List of dictionaries containing the search results
    :return: Dictionary with aggregated content
    """
    aggregated = {
        "total_items": len(data),
        "unique_authors": set(),
        "total_downloads": 0,
        "item_types": {"Models": 0, "Datasets": 0, "Spaces": 0}
    }

    for item in data:
        aggregated["unique_authors"].add(item.get("author", "Unknown"))
        aggregated["total_downloads"] += item.get("downloads", 0)
        
        if "modelId" in item:
            aggregated["item_types"]["Models"] += 1
        elif "dataset" in item.get("id", ""):
            aggregated["item_types"]["Datasets"] += 1
        else:
            aggregated["item_types"]["Spaces"] += 1

    aggregated["unique_authors"] = len(aggregated["unique_authors"])
    
    return aggregated

with gr.Blocks() as demo:
    gr.Markdown("## Search the Hugging Face Hub")
    with gr.Row():
        search_query = gr.Textbox(label="Search Query", value="awacke1")
        search_type = gr.Radio(["Models", "Datasets", "Spaces"], label="Search Type", value="Models")
        search_button = gr.Button("Search")
    results_html = gr.HTML(label="Search Results")
    metadata_output = gr.Textbox(label="Metadata", lines=10)
    aggregated_output = gr.JSON(label="Aggregated Content")

    def search_and_aggregate(query, search_type):
        df = search_hub(query, search_type)
        aggregated = SwarmyTime(df.to_dict('records'))
        html_results = display_results(df)
        return html_results, aggregated

    search_button.click(search_and_aggregate, inputs=[search_query, search_type], outputs=[results_html, aggregated_output])

demo.launch(debug=True)