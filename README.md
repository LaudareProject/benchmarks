# LaudareDataset Benchmarking Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18922366.svg)](https://doi.org/10.5281/zenodo.18922366)

This project provides a framework for benchmarking various tools and models for Handwritten Text Recognition (HTR), Handwritten Music Recognition (HMR), and Layout Recognition tasks on the Laudare
Dataset. It supports experiments using 5-fold cross-validation, sequential learning, and
cross-manuscript setups.

## Setup

Install system dependencies:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- `curl` and `jq`

## Reproducibility

1. **Download data**: put it in the `./data` directory: [Link to Zenodo](https://zenodo.org/records/18922615)
2. **Run the experiments**: to reproduce my experiments, just run the main script:

```bash
./experiments.sh data/I-Ct_91
./experiments.sh data/I-Fn_BR_18
```

You can also run more specific experiments (see the Help file) and/or add your own framework.

3. **Review Results**:
    - Benchmark results for the single fold run will be stored in subdirectories under `benchmarking/results/diplomatic/fold_test_0/`.
    - Log files for each tool and task will also be available.

## Use this Dataset with your Model/Framework

### Quick Start

To add your custom model:

1. Create `benchmarking/train_test/train_test_yourmodel.py` using the template below
2. Add model configurations to `benchmarking/models.json`
3. Test with: `python -m benchmarking.run_single_fold_benchmark --framework yourmodel --model-name "yourmodelsize" --debug --task yourtask`

### Template

Have a look at [`train_test_yolo.py`](`https://github.com/LaudareProject/LaudareBenchmarks/blob/master/benchmarking/train_test/train_test_yolo.py`) and the other existing `train_test_*.py` scripts for fully working examples. Below is a self-explaining template.

```python

def train_test_yourmodel(
        args=None, # the arguments parsed from command line by run_single_fold_benchmark.py
        is_train_test_mode=None, # if True, no val_json is provided, do it yourself
        is_sequential=None, # if True, this function is running as a step in sequential learning
        output_dir=None, # where to save predictions (see format below)
        train_json=None, # the COCO-format JSON file for training data
        val_json=None, # the COCO-format JSON file for validation data
        test_json=None, # the COCO-format JSON file for test data, this is None when pre-training
        save_model_path=None, # where to save the trained model (if any)
        load_model_path=None, # path to a file containing a pretrained model (if any)
        model_to_finetune=None, # an identifier (e.g. model name or path) of a base model
        # this is the one configured in models.json and has lower priority than
        # load_model_path (i.e. if load_model_path is given, it must be
        # used, not this parameter)
):
  ##################################
  # some information that can be accessed are in args, e.g. args.data_dir, args.framework,
  # args.task, args.debug, args.model_index, args.sequential_step, args.pretrained_model,
  # args.edition
  # see `uv run -m benchmarking.run_single_fold_benchmark --help` for details
  ##################################
  # Use utils.path_json2pagexml if you need PageXML files as input
  ##################################
  if args.task == "ocr":
    # load your model, see below for recommended loading patterns
    model = load_your_model_function(model_name, base_model, load_model_path, args)

    # train your model here
    model = train_your_model(train_json, val_json, model)

    save_your_model_function(model, save_model_path)

    if test_json is not None:
      # then load the best model and run predictions on the test set
      # see below for the standardized output formats
      test_predictions = predict_your_model(test_json, model)

      # save predictions in the given directory with the proper naming scheme
      # you can use annotations.ann_handler.create_new_pagexml_file(...) to easily create
      # PageXML files
      save_your_predictions(test_predictions, output_dir)
```

### Input Data Format

The framework provides COCO-style and PageXML annotations.

Here is an example of our COCO-style JSON format:

```json
// Example from ocr_test.json
{
  "images": [
    {
      "id": 5,                    // Unique image identifier
      "width": 2083,              // Image dimensions
      "height": 2717,
      "file_name": "c003r.png"    // Relative path from image-root
    }
  ],
  "annotations": [
    {
      "bbox": [434, 358, 1524, 141],  // MANDATORY: [x, y, width, height]
      "image_id": 2,                   // MANDATORY: Links to images[id]
      "category_id": 6,                // MANDATORY: From categories[]
      "description": "text content",   // MANDATORY for OCR/OMR
      "id": 73,                        // Unique annotation ID
      "iscrowd": 0,                    // Optional
      "area": 214884,                  // Optional
      "segmentation": []               // Optional
    }
  ],
  "categories": [                    // MANDATORY: Category definitions
    {
      "id": 6,                       // Matches category_id in annotations
      "name": "line",                // Human-readable name
      "supercategory": "layout"      // Task grouping
    }
  ]
}
```

In the above, note that the only non-standard field is `description`, which contains the text or
music content for OCR/OMR tasks.

We also provide PageXML files. Since the function you need to implement receives the paths to json
files, you can find the list of PageXML files corresponding to any json file
with the function `benchmarking.utils.path_json2pagexml(...)`.

### Output Requirements

#### HTR/HMR Tasks

- **Format**: Text files (`{image_stem}.pred.txt`) or, alternatively, PageXML files.
- **Content**: Concatenated text predictions for all lines in image
- **Example**: `"predicted text for line 1 predicted text for line 2"`

For PageXML files, the framework parses text/music in the `TextLine > TextEquiv >
Unicode` tags.

#### Layout Detection

- **Format**: PageXML files (`{image_stem}.xml`)
- **Method**: Use `annotations.ann_handler.create_new_pagexml_file()`

### Model Configuration

Add to `benchmarking/models.json`:

```json
"yourmodel": {
  "ocr": {
    "small": "yourmodel-small-ocr",
    "base": "yourmodel-base-ocr",
    "large": "yourmodel-large-ocr"
  },
  "omr": {
    "default": "path/to/your/omr/model"
  },
  "layout": {
    "default": "path/to/your/layout/model"
  }
}
```

## Credits

LAUDARE PRoject -- <https://laudare.eu/>
ERC Advanced Grant (follow the link for details)

Framework by Federico Simonetta -- <https://federicosimonetta.eu.org>
