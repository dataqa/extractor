from collections import Counter
import csv
import io
from typing import Optional

from instructor import OpenAISchema
import pandas as pd
from pydantic import create_model
from pydantic.fields import FieldInfo
import streamlit as st

from constants import *


def get_open_ai_client(openai_api_key):
    import openai as client
    client.api_key = openai_api_key
    return client


def define_extractor_class(fields, extraction_summary):
    num_cols = len(fields)
    args = {"__base__": OpenAISchema}
    for i in range(num_cols):
        if fields[i]['type'] == "int":
            selected_type = Optional[int]
        else:
            selected_type = Optional[str]
        args[fields[i]["name"]] = (
            selected_type,
            FieldInfo(description=fields[i]["desc"]),
        )

    ExtractedDataClass = create_model("ExtractedDataClass", **args)
    ExtractedDataClass.__doc__ = extraction_summary
    return ExtractedDataClass


def process_file(openai_api_key, uploaded_file, output_file_path, fields, extraction_summary):
    ExtractedDataClass = define_extractor_class(fields, extraction_summary)
    client = get_open_ai_client(openai_api_key)
    count_extractions = Counter()
    if uploaded_file is not None:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        csv_reader = csv.reader(stringio)

        total_cost = 0

        with open(output_file_path, "w") as output:
            writer = csv.writer(output)
            example = {}
            column_names = []

            for line_ind, line in enumerate(csv_reader):
                if len(line) > 0:
                    text = line[0]

                    completion = client.ChatCompletion.create(
                        model=MODEL,
                        functions=[ExtractedDataClass.openai_schema],
                        function_call={"name": ExtractedDataClass.openai_schema["name"]},
                        messages=[
                            {
                                "role": "system",
                                "content": f"I'm going to ask for {ExtractedDataClass.__doc__}. Use {ExtractedDataClass.openai_schema['name']} to parse this data.",
                            },
                            {
                                "role": "user",
                                "content": text,
                            },
                        ],
                    )

                    prompt_tokens = completion.usage.prompt_tokens
                    completion_tokens = completion.usage.completion_tokens

                    if MODEL in MODEL_COSTS:
                        total_cost += (prompt_tokens * MODEL_COSTS[MODEL]["input"] + completion_tokens *
                                       MODEL_COSTS[MODEL]["output"]) / 1000

                    user_details = ExtractedDataClass.from_response(completion)
                    line_values = user_details.model_dump()

                    if line_ind == 0:
                        column_names = list(line_values.keys())
                        writer.writerow(column_names)
                        example = dict((k, line_values[k]) for k in column_names)
                        example["original_text"] = text

                    output_line = []
                    for col in column_names:
                        value = line_values[col]
                        if value:
                            count_extractions[col] += 1
                        output_line.append(value)
                    writer.writerow(output_line)

    return {"total_lines": line_ind + 1,
            "total_cost": total_cost,
            "example": example,
            "total_extractions": count_extractions}


def check_fields(fields):
    for field in fields:
        if not ((len(field["name"]) > 0) & (len(field["type"]) > 0) & (len(field["desc"]) > 0)):
            return False
    return True


def streamlit_app():
    st.title("Extract data from texts")

    with st.container():
        st.header('Config', divider='rainbow')

        st.write("We recommend running on a sample first to get an estimate of quality and cost of extraction.")

        openai_api_key = st.text_input("Enter a valid OpenAI API key:", type="password")
        uploaded_file = st.file_uploader("Choose a csv file", accept_multiple_files=False, type="csv")

        output_folder_path = st.text_input(
            "Enter the folder path where you want to save the results:", HOME + "/Downloads"
        )

        num_cols = st.number_input(
            "Number fields to extract", min_value=1, max_value=20, value=None
        )

    if num_cols:

        with st.form("my_form", border=False):

            st.header('Fields', divider='rainbow')

            extraction_summary = st.text_input(
                label="Brief summary of the data to be extracted"
            )

            fields = []
            for i in range(num_cols):
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        name_field = st.text_input(key=f"{i}_1", label="Field name")
                    with col2:
                        desc_field = st.text_input(key=f"{i}_2", label="Description field")
                    with col3:
                        type_field = st.selectbox(
                            key=f"{i}_3", label="Type of field", options=("str", "int")
                        )
                    fields.append(
                        {"name": name_field, "desc": desc_field, "type": type_field}
                    )

            # Every form must have a submit button.
            submitted = st.form_submit_button("Run extractors")

            if submitted:
                if uploaded_file:
                    if not check_fields(fields):
                        st.warning("The name, type and description of a field need to be filled.")
                    else:
                        output_file_path = f"{output_folder_path}/extracted_data.csv"
                        result = process_file(openai_api_key,
                                              uploaded_file,
                                              output_file_path,
                                              fields,
                                              extraction_summary)
                        with st.container(border=True):
                            st.subheader("Results")
                            if result["total_cost"] > 0:
                                st.write(
                                    f"Extracting values from {result['total_lines']} lines cost an estimated ${result['total_cost']:0.2f}.")
                            else:
                                st.write(
                                    f"Extracting values from {result['total_lines']} lines.")
                            st.write(f"Results written to {output_file_path}")
                            st.markdown("**Summary extractions**")
                            st.table(pd.DataFrame.from_records([result["total_extractions"]]))
                            if result["example"]:
                                st.markdown("**Example**")
                                st.table(pd.DataFrame.from_records([result["example"]]))

                else:
                    st.warning("Need to upload csv file")


if __name__ == "__main__":
    streamlit_app()
