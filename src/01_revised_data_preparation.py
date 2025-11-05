import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_TREATMENT_DATE = pd.to_datetime("2024-02-16")
TREATED_TAG = "tech"
UNTREATED_TAGS = ['economie', 'achterklap', 'Media_en_Cultuur', 'goed_nieuws']
UNIT_ID_COL = "ARTICLE_SHORT_ID"
DATE_COL = "EVENT_DATE"
TAG_COL = "TAG"
PRIMARY_OUTCOME = "DAILY_PAGEVIEWS"
MIN_COHORT_SIZE = 15
TAG_MAPPINGS = {
    'media': 'Media_en_Cultuur', 
    'muziek': 'Media_en_Cultuur', 
    'film': 'Media_en_Cultuur', 
    'cultuur-overig': 'Media_en_Cultuur', 
    'economie': 'economie', 
    'achterklap': 'achterklap', 
    'tech': 'tech',
    'tweakers': 'tweakers', 
    'goed_nieuws': 'goed_nieuws', 
    'geod_niuews': 'goed_nieuws'
}
MAIN_TAGS = ['economie', 'achterklap', 'tech', 'goed_nieuws', 'Media_en_Cultuur', 'tweakers']
URL_PATTERN = r'https://www\.nu\.nl/([^/]+)/'


# Spatial proximity scores (relative distance, 1 closes 6 farthest))
SPATIAL_SCORES = {
    'goed_nieuws': 1,
    'tech': 1,
    'achterklap': 3,
    'Media_en_Cultuur': 4,
    'economie': 6
}

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns: return name
    return None

def standardize_columns(df):
    print("  - Standardizing column names...")
    cols_to_standardize = {'URL': ['URL', 'url', 'Url'], 'EVENT_DATE': ['EVENT_DATE', 'event_date', 'Date', 'date'], 'ARTICLE_SHORT_ID': ['ARTICLE_SHORT_ID', 'article_short_id', 'article_id', 'ArticleId'], 'TITLE': ['TITLE', 'title', 'Title'], 'ORIGINAL_TAG': ['ORIGINAL_TAG', 'tag', 'Tag']}
    rename_map = {found_col: std_name for std_name, variations in cols_to_standardize.items() if (found_col := find_column(df, variations)) and found_col != std_name}
    if rename_map: df = df.rename(columns=rename_map)
    return df

def extract_and_map_tags(df):
    print("  - Extracting and mapping tags...")
    if 'ORIGINAL_TAG' not in df.columns: df['ORIGINAL_TAG'] = df['URL'].astype(str).str.extract(URL_PATTERN)
    df['TAG'] = df['ORIGINAL_TAG'].map(TAG_MAPPINGS).fillna(df['ORIGINAL_TAG'])
    return df[df['TAG'].isin(MAIN_TAGS)].copy()

def create_outcome_columns(df):
    print("  - Creating outcome variables...")
    all_traffic_cols = [c for c in df.columns if 'DAILY_PAGEVIEWS_' in c]
    for col in all_traffic_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if PRIMARY_OUTCOME not in df.columns: df[PRIMARY_OUTCOME] = df[all_traffic_cols].sum(axis=1)
    external_cols = [c for c in all_traffic_cols if any(x in c for x in ['SEARCH', 'GOOGLE_NEWS', 'FACEBOOK', 'TWITTER', 'INSTAGRAM', 'LINKEDIN', 'SOCIAL_OTHER'])]
    loyal_cols = [c for c in all_traffic_cols if any(x in c for x in ['DIRECT', 'EMAIL'])]
    df['traffic_external_discovery'] = df[external_cols].sum(axis=1)
    df['traffic_loyal_audience'] = df[loyal_cols].sum(axis=1)
    return df

def preprocess_text(text_series):
    return text_series.fillna("").astype(str).str.lower()

def get_sentence_embeddings(texts, model_name='paraphrase-MiniLM-L6-v2', batch_size=32):
    print("  - Generating title embeddings...")
    processed_texts = preprocess_text(texts).tolist()
    return SentenceTransformer(model_name).encode(processed_texts, batch_size=batch_size, show_progress_bar=True)

def engineer_features_and_winsorize(df):
    print("\n--- Running Feature Engineering & Outlier Management ---")
    pre_treatment_data = df[df[DATE_COL] < BASE_TREATMENT_DATE]
    outcome_vars = [PRIMARY_OUTCOME, 'traffic_external_discovery', 'traffic_loyal_audience']
    for var in outcome_vars:
        if var in pre_treatment_data.columns and not pre_treatment_data[var].empty:
            threshold = pre_treatment_data[var].quantile(0.95)
            df[var] = df[var].clip(upper=threshold)
            print(f"  - '{var}' winsorized at 95th percentile: {threshold:.2f}")
    return df

def detect_treatment_cohorts_by_onset(df, target_vertical, launch_date, onset_threshold, persistence):
    print(f"\n--- Running Method: Effect Onset Detection (Target: {target_vertical}, Threshold: {onset_threshold}) ---")
    treatment_cohorts = {}
    
    # This is the key change: Use the 'target_vertical' argument
    article_ids = df[df[TAG_COL] == target_vertical][UNIT_ID_COL].unique()
    
    for article in tqdm(article_ids, desc=f"Detecting onsets for {target_vertical}"):
        article_data = df[df[UNIT_ID_COL] == article].sort_values(DATE_COL).set_index(DATE_COL)
        pre_data = article_data[article_data.index < launch_date]
        if len(pre_data) < 7: continue
        baseline = pre_data[PRIMARY_OUTCOME].mean()
        post_data = article_data[article_data.index >= launch_date].copy()
        if len(post_data) >= 7:
            post_data["rolling_mean"] = post_data[PRIMARY_OUTCOME].rolling(window=3, center=True, min_periods=1).mean()
            post_data['below_threshold'] = post_data['rolling_mean'] < (baseline * onset_threshold)
            post_data['block'] = (post_data['below_threshold'].diff() != 0).cumsum()
            persistent_blocks = post_data[post_data['below_threshold']].groupby('block').filter(lambda x: len(x) >= persistence)
            if not persistent_blocks.empty:
                treatment_cohorts[article] = persistent_blocks.index.min()
    print(f"  - Initial effect onset detected for {len(treatment_cohorts)} articles in {target_vertical}.")
    return {article: pd.to_datetime(date).to_period("W").start_time for article, date in treatment_cohorts.items()}

def create_artificial_stagger(df, target_vertical, launch_date):
    print(f"\n--- Running Method: Artificial Stagger (Target: {target_vertical}) ---")
    
    # This is the key change: Use the 'target_vertical' argument
    treated_articles = df[df[TAG_COL] == target_vertical][UNIT_ID_COL].unique()
    
    np.random.seed(42)
    assignments = np.random.randint(0, 4, size=len(treated_articles))
    return {article: launch_date + pd.Timedelta(days=int(assignments[i] * 7)) for i, article in enumerate(treated_articles)}

def consolidate_small_cohorts(cohort_map, min_size):
    if not cohort_map: return {}
    cohort_df = pd.DataFrame.from_dict(cohort_map, orient="index", columns=["cohort_date"])
    cohort_counts = cohort_df["cohort_date"].value_counts()
    small_cohorts = cohort_counts[cohort_counts < min_size].index.tolist()
    if not small_cohorts: return cohort_map
    valid_cohorts = sorted(cohort_counts[cohort_counts >= min_size].index.tolist())
    updated_cohort_map = cohort_map.copy()
    for unit, date in cohort_map.items():
        if date in small_cohorts and valid_cohorts:
            time_diffs = [(c, abs(c - date)) for c in valid_cohorts]
            updated_cohort_map[unit] = min(time_diffs, key=lambda x: x[1])[0]
    return updated_cohort_map

def create_thematic_tiers_cohorts(df):
    print("\n--- Assigning Thematic Tiers ---")
    tech_articles = df[df[TAG_COL] == TREATED_TAG].copy() # Use TREATED_TAG ("tech")
    if 'similarity_to_tweakers_title' not in tech_articles.columns or tech_articles['similarity_to_tweakers_title'].isnull().all():
        raise ValueError("Similarity column missing or all null.")
    tech_articles['thematic_tier'] = pd.qcut(tech_articles['similarity_to_tweakers_title'], q=4, labels=False, duplicates='drop')
    tech_articles['thematic_tier'] = 3 - tech_articles['thematic_tier']
    return pd.Series(tech_articles.thematic_tier.values, index=tech_articles[UNIT_ID_COL]).to_dict()

def create_and_save_tier_datasets(df, cohort_map, method_name, processed_dir):
    print(f"\n--- Generating Tiered Datasets for '{method_name}' ---")
    all_tiers = sorted(list(set(cohort_map.values())))
    if len(all_tiers) <= 1:
        print(f"  - WARNING: All treated articles fall into a single tier: {all_tiers}.")
    
    # Use TREATED_TAG ("tech")
    control_df = df[df[TAG_COL] != TREATED_TAG].copy() 
    
    for tier in all_tiers:
        tier_article_ids = {k for k, v in cohort_map.items() if v == tier}
        if not tier_article_ids: continue
        tier_df_treated = df[df[UNIT_ID_COL].isin(tier_article_ids)].copy()
        combined_df = pd.concat([tier_df_treated, control_df], ignore_index=True)
        min_date = combined_df[DATE_COL].min()
        combined_df['gname'] = np.where(combined_df[UNIT_ID_COL].isin(tier_article_ids), 2, 0)
        combined_df["time_period"] = (combined_df[DATE_COL] - min_date).dt.days + 1
        output_path = processed_dir / f"cs_data_{method_name}_tier_{tier}.parquet"
        cols_to_keep = [UNIT_ID_COL, "time_period", "gname", PRIMARY_OUTCOME, TAG_COL]
        combined_df[[col for col in cols_to_keep if col in combined_df.columns]].to_parquet(output_path, index=False)
        print(f"  - Tier {tier}: Saved data for {len(tier_article_ids)} articles to {output_path}")

def generate_and_save_meta_data(df, processed_dir):
    """
    Calculates vertical-level thematic similarity and saves it for meta-analysis in R.
    """
    print("\n--- Generating Meta-Analysis Inputs ---")
    if 'similarity_to_tweakers_title' not in df.columns:
        print("  - WARNING: Similarity column not found. Skipping meta-analysis data generation.")
        return

    # Get all verticals from the main df (excluding the new 'tweakers' vertical)
    all_verticals = df[df['TAG'].isin(MAIN_TAGS) & (df['TAG'] != 'tweakers')]
    
    # Calculate and save the average similarity for each vertical
    all_vertical_similarity = all_verticals.groupby(TAG_COL)['similarity_to_tweakers_title'].mean().reset_index()
    all_vertical_similarity.rename(
        columns={TAG_COL: 'Vertical', 'similarity_to_tweakers_title': 'avg_thematic_similarity'}, 
        inplace=True
    )

    output_path = processed_dir / "vertical_thematic_scores.csv"
    all_vertical_similarity.to_csv(output_path, index=False)
    print(f"✅ Successfully saved vertical thematic scores to: {output_path}")
    print("  - Thematic Scores Preview:")
    print(all_vertical_similarity.head().to_string(index=False))

def main(args):
    PROCESSED_DATA_DIR = Path("./data/processed")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"--- 1. Loading and Initial Cleaning ---")
    df = pd.read_csv(args.input_file)
    df = standardize_columns(df)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df = df.dropna(subset=[DATE_COL, 'URL', UNIT_ID_COL])
    df = extract_and_map_tags(df)
    df = create_outcome_columns(df)
    df_filtered = df.groupby(UNIT_ID_COL).filter(lambda x: (x[DATE_COL] < BASE_TREATMENT_DATE).any() and (x[DATE_COL] >= BASE_TREATMENT_DATE).any()).copy()
    print(f"  - Filtered for articles present before & after launch: {df_filtered[UNIT_ID_COL].nunique()} articles.")

    if args.method == 'thematic_tiers':
        print("\n--- 2. Generating Embeddings & Similarities ---")
        df_filtered['title_embedding'] = list(get_sentence_embeddings(df_filtered['TITLE']))
        tweakers_df = df_filtered[df_filtered['TAG'] == 'tweakers']
        if not tweakers_df.empty:
            initial_wave = tweakers_df[(tweakers_df[DATE_COL] >= BASE_TREATMENT_DATE) & (tweakers_df[DATE_COL] < BASE_TREATMENT_DATE + pd.Timedelta(days=7))]
            if not initial_wave.empty:
                prototype = np.mean(np.vstack(initial_wave['title_embedding']), axis=0).reshape(1, -1)
                df_filtered['similarity_to_tweakers_title'] = df_filtered['title_embedding'].apply(lambda emb: cosine_similarity(emb.reshape(1, -1), prototype)[0, 0] if isinstance(emb, np.ndarray) else np.nan)
                generate_and_save_meta_data(df_filtered, PROCESSED_DATA_DIR)

    df_clean = df_filtered[df_filtered['TAG'] != 'tweakers'].copy()
    df_processed = engineer_features_and_winsorize(df_clean)

    print("  - Saving total article counts by vertical...")
    vertical_counts = df_processed.groupby(TAG_COL)[UNIT_ID_COL].nunique().reset_index()
    vertical_counts.rename(columns={TAG_COL: 'Vertical', UNIT_ID_COL: 'Total_Articles_Platform'}, inplace=True)
    vertical_counts.to_csv(PROCESSED_DATA_DIR / "vertical_article_counts.csv", index=False)

    if args.method == "thematic_tiers":
        cohort_map = create_thematic_tiers_cohorts(df_processed)
        create_and_save_tier_datasets(df_processed, cohort_map, args.method, PROCESSED_DATA_DIR)
        print("\n✅ Tiered mechanism datasets generated successfully.")
        
    elif args.method == "naive_launch":
        print("\n--- Running Method: Truly Naive (Uniform Launch Date) ---")
        final_df = df_processed.copy()
        min_date = final_df[DATE_COL].min()
        launch_gname = (BASE_TREATMENT_DATE - min_date).days + 1
        final_df['gname'] = np.where(final_df[TAG_COL] == TREATED_TAG, launch_gname, 0)
        final_df["time_period"] = (final_df[DATE_COL] - min_date).dt.days + 1
        output_path = PROCESSED_DATA_DIR / "cs_data_naive_launch.parquet"
        cols_to_keep = [UNIT_ID_COL, "time_period", "gname", PRIMARY_OUTCOME, 'traffic_loyal_audience', 'traffic_external_discovery', TAG_COL]
        final_df = final_df[[col for col in cols_to_keep if col in final_df.columns]]
        final_df.to_parquet(output_path, index=False)
        print(f"\n✅ Successfully saved truly naive data to: {output_path}")


    elif args.method == "effect_onset" and args.placebo_type == 'none':
        print("\n--- Running Method: Master Effect Onset Detection ---")
        analysis_df = df_processed.copy()
        all_cohorts = {}
        all_verticals = ['tech', 'economie', 'achterklap', 'Media_en_Cultuur', 'goed_nieuws']
        
        for vertical in all_verticals:
            vertical_cohorts = detect_treatment_cohorts_by_onset(
                analysis_df, 
                vertical,
                BASE_TREATMENT_DATE, 
                args.onset_threshold, 
                args.persistence
            )
            all_cohorts.update(vertical_cohorts)
        
        consolidated_cohort_map = consolidate_small_cohorts(all_cohorts, MIN_COHORT_SIZE)
        
        final_df = analysis_df.copy()
        min_date = final_df[DATE_COL].min()
        if consolidated_cohort_map:
            gname_map = {unit: (date - min_date).days + 1 for unit, date in consolidated_cohort_map.items()}
            final_df['gname'] = final_df[UNIT_ID_COL].map(gname_map).fillna(0).astype(int)
        else:
            final_df['gname'] = 0
            
        final_df["time_period"] = (final_df[DATE_COL] - min_date).dt.days + 1
        
        # --- Create the single master filename ---
        output_filename = f"cs_data_{args.method}_{int(args.onset_threshold*100)}pct_MASTER.parquet"
        output_path = PROCESSED_DATA_DIR / output_filename
        
        cols_to_keep = [UNIT_ID_COL, "time_period", "gname", PRIMARY_OUTCOME, 'traffic_loyal_audience', 'traffic_external_discovery', TAG_COL, 'ORIGINAL_TAG']
        final_df = final_df[[col for col in cols_to_keep if col in final_df.columns]]
        final_df.to_parquet(output_path, index=False)
        print(f"\n✅ Successfully saved MASTER onset data to: {output_path}")


    else:
        # This block now ONLY handles 'artificial_stagger' and 'pre_period' placebos
        analysis_df = df_processed.copy()
        
        launch_date = BASE_TREATMENT_DATE
        if args.placebo_type == 'pre_period':
            launch_date -= pd.Timedelta(days=28)
            analysis_df = analysis_df[analysis_df[DATE_COL] < BASE_TREATMENT_DATE - pd.Timedelta(days=14)].copy()
        
        if args.method == "effect_onset": # This is now only for pre_period
            cohort_map = detect_treatment_cohorts_by_onset(
                analysis_df, args.vertical, launch_date, 
                args.onset_threshold, args.persistence
            )
        else: # artificial_stagger
            cohort_map = create_artificial_stagger(
                analysis_df, args.vertical, launch_date
            )
        
        consolidated_cohort_map = consolidate_small_cohorts(cohort_map, MIN_COHORT_SIZE)
        
        final_df = analysis_df.copy()
        min_date = final_df[DATE_COL].min()
        if consolidated_cohort_map:
            gname_map = {unit: (date - min_date).days + 1 for unit, date in consolidated_cohort_map.items()}
            final_df['gname'] = final_df[UNIT_ID_COL].map(gname_map).fillna(0).astype(int)
        else:
            final_df['gname'] = 0
            
        final_df["time_period"] = (final_df[DATE_COL] - min_date).dt.days + 1
        
        # Filename logic is unchanged, will create the correct placebo/stagger files
        output_filename = f"cs_data_{args.method}"
        if args.method == 'effect_onset': 
            output_filename += f"_{int(args.onset_threshold*100)}pct"
        output_filename += f"_{args.vertical}" 
        if args.placebo_type == 'pre_period': 
            output_filename += f"_placebo_pre_period"
        
        output_path = PROCESSED_DATA_DIR / f"{output_filename}.parquet"
        
        cols_to_keep = [UNIT_ID_COL, "time_period", "gname", PRIMARY_OUTCOME, 'traffic_loyal_audience', 'traffic_external_discovery', TAG_COL, 'ORIGINAL_TAG']
        final_df = final_df[[col for col in cols_to_keep if col in final_df.columns]]
        final_df.to_parquet(output_path, index=False)
        print(f"\n✅ Successfully saved placebo/stagger data to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified data preparation for DiD analysis.")
    parser.add_argument("--input_file", type=str, default="./data/raw/raw_data.csv", help="Path to the raw input CSV file.")
    parser.add_argument("--method", type=str, required=True, choices=["effect_onset", "artificial_stagger", "thematic_tiers", "naive_launch"], help="Method for creating analysis datasets.")
    parser.add_argument("--vertical", type=str, default="tech", help="Target vertical to build cohorts for (e.g., 'tech', 'economie'). Default is 'tech'.")
    parser.add_argument("--onset_threshold", type=float, default=0.8, help="Threshold for effect onset detection (e.g., 0.8 for a 20% drop).")
    parser.add_argument("--persistence", type=int, default=3, help="Number of consecutive days below threshold to confirm onset.")
    parser.add_argument("--placebo_type", type=str, default="none", choices=["none", "pre_period"], help="Type of general placebo test to run.")
    args = parser.parse_args()
    
    # It sets the global TREATED_TAG *only* for the 'tech-specific' methods
    if args.method not in ["thematic_tiers", "naive_launch"]:
        # For onset/stagger, TREATED_TAG is not used by the main functions,
        # but this is harmless. The 'args.vertical' is passed directly.
        pass 
    else:
        # For naive and thematic, we *always* use the default "tech"
        # This correctly sets the global TREATED_TAG for those helper functions.
        TREATED_TAG = "tech"
        
    main(args)