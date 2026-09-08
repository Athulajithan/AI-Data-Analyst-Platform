import os
import argparse
import sys
from src.data_engine.loader import DataLoader
from src.data_engine.profiler import DataProfiler
from src.data_engine.cleaner import DataCleaner
from src.data_engine.validator import DataValidator
from src.data_engine.eda import EDAEngine
from src.data_engine.statistical_analyzer import StatisticalAnalyzer
from src.data_engine.sql_engine import SQLEngine
from src.data_engine.viz_engine import VisualizationEngine
from src.data_engine.kpi_engine import KPIEngine
from src.data_engine.ml_engine import MLEngine
from src.agent_core.insight_generator import InsightGenerator
from src.agent_core.recommendation_engine import RecommendationEngine
from src.agent_core.report_builder import ReportBuilder
from src.exports.exporter import DataExporter

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="AI Data Analyst Agent CLI Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to input dataset file (CSV, XLSX, Parquet, JSON)")
    parser.add_argument("--output", "-o", default="output_report", help="Directory where reports and cleaned data will be saved")
    parser.add_argument("--target", "-t", default=None, help="Optional target column for ML baseline training")
    
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    print(f"\n==================================================")
    print(f"   AI DATA ANALYST AGENT: AUTOMATED PIPELINE     ")
    print(f"==================================================")
    print(f"📂 Loading input file: {input_path}")

    # 1. Ingestion
    df_original, metadata = DataLoader.load_file(input_path)
    dataset_name = os.path.splitext(os.path.basename(input_path))[0]
    print(f"✅ Loaded {metadata['num_rows']} rows, {metadata['num_cols']} columns.")

    # 2. Profiling & Quality Scoring
    print("🔍 Profiling dataset & calculating Data Quality Score...")
    profiling = DataProfiler.profile_dataset(df_original, metadata)
    print(f"⭐ Data Quality Score: {profiling['data_quality_score']} / 100")

    # 3. Cleaning
    print("🧹 Executing non-destructive cleaning pipeline...")
    cleaner = DataCleaner(df_original, metadata)
    df_cleaned, transformation_log = cleaner.run_full_cleaning_pipeline()
    print(f"✅ Cleaning completed ({len(transformation_log)} log entries recorded).")

    # 4. Validation
    print("⚖️ Validating pre vs post dataset metrics...")
    validation = DataValidator.validate_cleaning(df_original, df_cleaned, transformation_log)

    # 5. EDA & KPIs
    print("📊 Computing Exploratory Data Analysis & Business KPIs...")
    eda_results = EDAEngine.run_eda(df_cleaned, metadata)
    kpi_results = KPIEngine.calculate_kpis(df_cleaned, metadata)

    # 6. Statistical Analysis
    print("📐 Running statistical hypothesis tests (t-tests, Chi2, ANOVA)...")
    stat_results = StatisticalAnalyzer.run_tests(df_cleaned, metadata)

    # 7. SQL Engine & Recommended Queries
    print("💾 Initializing in-memory SQL Engine...")
    sql_engine = SQLEngine(df_cleaned, table_name=dataset_name)
    sql_queries = sql_engine.generate_recommended_queries(metadata)
    sql_engine.close()

    # 8. Machine Learning Extension
    ml_results = None
    target_candidate = args.target or (metadata["column_buckets"]["target_candidates"][0] if metadata["column_buckets"]["target_candidates"] else None)
    if target_candidate:
        print(f"🤖 Training baseline ML model on target column '{target_candidate}'...")
        ml_results = MLEngine.train_baseline_model(df_cleaned, target_col=target_candidate)

    # 9. Chart Recommendations
    print("🎨 Generating chart recommendations...")
    chart_recs = VisualizationEngine.recommend_charts(df_cleaned, metadata)

    # 10. AI Insights & Recommendations Synthesis
    print("🧠 Synthesizing 4-Level insights & prioritized recommendations...")
    insights = InsightGenerator.generate_insights(metadata, profiling, validation, eda_results, stat_results, kpi_results, ml_results)
    recommendations = RecommendationEngine.generate_recommendations(profiling, validation, eda_results, stat_results, kpi_results)

    # 11. Report Generation
    print("📝 Compiling standardized executive report...")
    report_md = ReportBuilder.build_markdown_report(
        dataset_name=dataset_name,
        metadata=metadata,
        profiling=profiling,
        validation=validation,
        transformation_log=transformation_log,
        eda_results=eda_results,
        stat_results=stat_results,
        kpi_results=kpi_results,
        chart_recs=chart_recs,
        insights=insights,
        recommendations=recommendations,
        ml_results=ml_results
    )

    # 12. Export Artifacts
    print(f"📦 Exporting cleaned files, report & HTML dashboard to: {output_dir}")
    exported = DataExporter.export_all(
        output_dir=output_dir,
        df_cleaned=df_cleaned,
        report_md=report_md,
        sql_queries=sql_queries,
        dataset_name=dataset_name,
        metadata=metadata,
        profiling=profiling,
        kpi_results=kpi_results,
        insights=insights
    )

    print("\n==================================================")
    print("✅ ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================")
    for k, v in exported.items():
        print(f"  • {k}: {v}")
    print("\n")

if __name__ == "__main__":
    main()
