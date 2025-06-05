# AI Dataset Labeler

A comprehensive tool for creating training datasets for AI models. Supports multiple export formats including YOLOv8, COCO, Pascal VOC, and custom JSON formats. Now includes **advanced auto-annotation capabilities** using OpenCV template matching and OpenAI Vision API!

## Features

### Core Functionality
- **Video Frame Annotation**: Load videos and annotate objects frame by frame
- **Multiple Export Formats**: YOLOv8, COCO, Pascal VOC, Custom JSON, MoViNet
- **Region Management**: Draw, edit, and label bounding boxes
- **Custom Labels**: Create and manage custom label sets
- **Clip Labeling**: Support for video clip annotation (MoViNet format)

### 🆕 Advanced Auto-Annotation Features

#### 1. OpenCV Template Matching
- **Create Templates**: Select any user-annotated region to create a template
- **Automatic Detection**: Find similar objects across frames using template matching
- **Confidence Thresholds**: Adjustable matching sensitivity
- **Template Management**: Save, remove, and organize templates
- **Manual Review**: All auto-annotations require user acceptance

#### 2. OpenAI Vision API Integration
- **AI-Powered Analysis**: Send frames to OpenAI's vision models for intelligent annotation
- **Structured Outputs**: Get precise bounding boxes with confidence scores
- **Custom Labels**: Provide preferred labels to guide the AI
- **Detailed Descriptions**: Receive explanatory text for each detection
- **Category Classification**: Automatic object categorization

#### 3. Smart Workflow
- **Pending Annotations**: Review all auto-generated annotations before accepting
- **Individual Acceptance**: Accept or reject annotations one by one
- **Batch Operations**: Accept or reject all pending annotations at once
- **Visual Indicators**: Clear UI indicators for OpenCV vs AI annotations
- **Confidence Display**: See confidence scores for all detections

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- PyQt5

### Install Dependencies
```bash
pip install -r requirements.txt
```

### For OpenAI Auto-Annotation (Optional)
1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the environment variable:
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export OPENAI_API_KEY=your_api_key_here
   ```
3. Or set it directly in the application using the "Set API Key" button

## Usage

### Basic Usage
1. **Launch the application**:
   ```bash
   python main.py
   ```

2. **Load a video** using the "Load Video" button

3. **Annotate manually**:
   - Draw bounding boxes by clicking and dragging
   - Add labels using the Label Manager
   - Navigate frames using controls or keyboard shortcuts

### Auto-Annotation Workflow

#### Using OpenCV Template Matching
1. **Create Templates**:
   - Manually annotate a few examples of objects you want to detect
   - Select an annotated region
   - Go to "Auto Annotation" tab → "OpenCV Template Matching"
   - Click "Create Template from Selected Region"
   - Provide a descriptive label

2. **Find Matches**:
   - Navigate to any frame
   - Click "Find Template Matches"
   - Review results in "Pending Annotations" tab

3. **Review and Accept**:
   - Switch to "Pending Annotations" tab
   - Review each detected object
   - Accept or reject individual annotations
   - Or use "Accept All" / "Reject All" for batch operations

#### Using OpenAI Vision API
1. **Setup API Key**:
   - Go to "Auto Annotation" tab → "OpenAI Vision"
   - Click "Set API Key" and enter your OpenAI API key

2. **Configure (Optional)**:
   - Add custom labels in the text field (comma-separated)
   - Example: "person, car, building, traffic light"

3. **Analyze Frame**:
   - Navigate to any frame
   - Click "Analyze Frame with AI"
   - Wait for processing (sends frame to OpenAI)

4. **Review Results**:
   - Switch to "Pending Annotations" tab
   - Review AI-generated annotations with descriptions
   - Accept or reject as needed

### Keyboard Shortcuts
- **Space**: Play/Pause video
- **Left/Right Arrow**: Previous/Next frame
- **Shift + Left/Right**: Skip 10 frames
- **Delete**: Delete selected region
- **Ctrl + A**: Select all regions
- **Escape**: Clear selection
- **Ctrl + O**: Load video
- **Ctrl + S**: Save current frame
- **Ctrl + E**: Export dataset

## Export Formats

### YOLOv8
- Images in `/images/` folder
- Labels in `/labels/` folder with normalized coordinates
- `classes.txt` file with label mappings

### COCO Format
- `annotations.json` with COCO-compatible structure
- Images in `/images/` folder

### Pascal VOC
- XML annotation files in `/Annotations/` folder
- Images in `/JPEGImages/` folder

### Custom JSON
- Complete dataset in single JSON file
- Includes frame labels, regions, and metadata

### MoViNet
- Video clips with action labels
- Suitable for video action recognition

## Testing Auto-Annotation

Run the test script to verify functionality:
```bash
python test_auto_annotation.py
```

This will:
- Test OpenCV template matching with synthetic data
- Test OpenAI integration (if API key is set)
- Generate test result images

## File Structure
```
AILabeler/
├── main.py                          # Application entry point
├── ai_labeler.py                     # Main application window
├── video_player.py                   # Video playback and annotation UI
├── label_manager.py                  # Label management widget
├── clip_manager.py                   # Video clip management
├── region_selector.py                # Region selection utilities
├── data_exporter.py                  # Export functionality
├── opencv_auto_annotator.py          # OpenCV template matching
├── openai_auto_annotator.py          # OpenAI Vision API integration
├── auto_annotation_manager.py        # Auto-annotation UI management
├── test_auto_annotation.py           # Test suite for auto-annotation
├── requirements.txt                  # Python dependencies
└── labeled_data/                     # Output directory
    ├── export_yolov8/
    ├── export_coco/
    ├── export_pascal_voc/
    └── export_custom_json/
```

## Tips for Best Results

### OpenCV Template Matching
- Create templates from clear, well-defined objects
- Use multiple templates for objects that appear at different scales/angles
- Adjust confidence thresholds based on your needs
- Templates work best for objects with consistent appearance

### OpenAI Vision API
- Provide clear, well-lit frames for best results
- Use custom labels to guide the AI toward your specific objects
- Review confidence scores - lower scores may need manual verification
- Consider cost - each API call has associated usage fees

### General Workflow
- Start with manual annotations to establish ground truth
- Use auto-annotation to speed up similar objects/scenes
- Always review auto-generated annotations before final export
- Combine both methods for comprehensive coverage

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool.

## License

This project is open source. Please check the license file for details. 