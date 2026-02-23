import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)
np.random.seed(42)

# Increased to 1000 students because predicting 10 different grades requires more data!
num_students = 1000 

# 1. Generate realistic marks based on DIU Assessment Scheme
attendance = np.random.randint(2, 8, num_students)        # Out of 7
class_test = np.random.randint(5, 16, num_students)       # Out of 15 (CT1, CT2, CT3)
assignment = np.random.randint(2, 6, num_students)        # Out of 5
presentation = np.random.randint(4, 9, num_students)      # Out of 8
midterm = np.random.randint(10, 26, num_students)         # Out of 25

# 2. Simulate the Semester Final Exam (Out of 40)
# We base this on how well they did in the midterm, plus some random real-world noise
final_exam_base = (midterm / 25) * 40 
noise = np.random.normal(0, 6, num_students)
final_exam = np.clip(np.round(final_exam_base + noise), 0, 40) 

# 3. Calculate Total Mark (Out of 100)
total_marks = attendance + class_test + assignment + presentation + midterm + final_exam

# 4. Map Total Marks to DIU Grading Scale
def get_diu_grade(mark):
    if mark >= 80: return 'A+'
    elif mark >= 75: return 'A'
    elif mark >= 70: return 'A-'
    elif mark >= 65: return 'B+'
    elif mark >= 60: return 'B'
    elif mark >= 55: return 'B-'
    elif mark >= 50: return 'C+'
    elif mark >= 45: return 'C'
    elif mark >= 40: return 'D'
    else: return 'F'

grades = [get_diu_grade(m) for m in total_marks]

# 5. Create DataFrame
# Notice we DO NOT include the 'Semester Final Examination' or 'Total Mark' in the final dataset.
# We want the AI to predict the Grade based ONLY on the Midterm and continuous assessments!
df = pd.DataFrame({
    'Attendance': attendance,
    'Class_Test': class_test,
    'Assignment': assignment,
    'Presentation': presentation,
    'Midterm': midterm,
    'Grade': grades
})

df.to_csv('data/raw_data.csv', index=False)
print("DIU Customized Dataset generated successfully!")