def check_for_leakage(df1, df2, patient_col):
    df1_patients_unique = set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)
    patients_in_both_groups = list(df1_patients_unique.
                                   intersection((df2_patients_unique)))
    leakage = len(patients_in_both_groups) > 0
    return leakage

if __name__ == "__main__":
    import pandas as pd
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print(check_for_leakage(train_df, test_df, "patient_id"))