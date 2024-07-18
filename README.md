This is the beginning of the ANIMATE (Animal Image Matching and Tagging Environment) program


To run the program, you can use the following command in your terminal or command prompt:
 
1. This command will process all images in the `data/images` directory.
```
python src/main.py --type images
```

2. If you want to filter for a specific species, such as "elephant," you can add the `--filter` argument:

```
python src/main.py --type images --filter elephant
```

If you have any additional tasks, such as fetching data from GBIF or retraining the model, you can include the respective arguments:

- To fetch data from GBIF:

```
python src/main.py --fetch
```

- To retrain the model with corrected data:

```
  python src/main.py --retrain
```

Combining arguments (e.g., fetching data, processing images, and filtering for elephants):

```sh
python src/main.py --fetch --type images --filter elephant
```


COMPLETED TASKS
- Image processing
- Video processing (too heavy for my laptop but it works)
- Recognizing multiple species
- Filtering and creating directory of input images


Next Steps
So far so good, now how can I create a function that makes corrections based on the observations. I can see some images have been labeled in correctly. So we do want to be able to adjust that. After adjustments, I want to program to lean and retain that information for future images and understand.



PENDING TASKS
- filter by color and shape


- Prediction Function

- Manual Tagging and Correction
Objective: Allow users to manually correct model predictions and update the dataset. This can used to correct any errors in the results.

Steps:
1. Display Image and Prediction: Show the image and the predicted label to the user.

2. Accept User Input: Allow the user to correct the label if needed.

3. Update Dataset: Save the corrected label to the dataset for future training.

- Counting and Tracking
Objective: Count occurrences of species and track their presence over time.
Steps:
1. Initialize Counters: Create a counter for each species.
2. Update Counters: Increment counters based on model predictions.
3. Track Over Time: Record timestamps and locations of detections for tracking.

- User Options for Analysis (Filtering, Data analysis)
Objective: Provide users with options to analyze data, such as counting occurrences, tracking species, and manual tagging.

Steps:
Create Menu: Provide a menu for users to select different analysis options.
Implement Options: Implement functions for each option (e.g., counting, tracking, tagging).



- INTEGRATE OPENAI MODEL (OPTIONAL - AND COSTS MONEY)
Objective: Use advanced models from OpenAI for more complex tasks.

Steps:

API Integration: Use OpenAI's API to access their models.
Make Predictions: Send image data to the OpenAI model and receive predictions.
Process Results: Integrate OpenAI's predictions into your existing workflow.

LOOK INTO
- megadetector for animal species
