

from ast import Dict
import re
import copy
import os
import pandas as pd
from typing import Optional, Tuple

from sklearn.preprocessing import MinMaxScaler




class GazePointProcessor:

    def scale_fixations_trial(self, fixations_trial, x_screen, y_screen):
        """
        Scale trial fixations to screen size
        """
        
        fixations_trial["x"] = fixations_trial["x"] * x_screen
        fixations_trial["y"] = fixations_trial["y"] * y_screen
        return fixations_trial
    

    def _find_time_column_name(self, df: pd.DataFrame) -> str:
        pattern = r"TIME\("

        for col in df.columns.tolist():
            if re.search(pattern, col):
                return col
        raise ValueError("No TIME column found in DataFrame")

    def preprocess_fixations(self, fixations: pd.DataFrame) -> pd.DataFrame:
        # better to use: BPOGX, BPOGY
        
        time_column_name = self._find_time_column_name(fixations)
        fixations = fixations.rename(
            columns={"FPOGX": "x", "FPOGY": "y", "FPOGD": "duration", "FPOGID": "ID", "USER": "trial", time_column_name: "TIME"}
        )
        fixations["pupil_r"] = round(
            fixations["RPD"] * fixations["RPS"], 4
        )
        fixations["pupil_l"] = round(
            fixations["LPD"] * fixations["LPS"], 4
        )
        fixations = fixations[["ID", "TIME", "x", "y", "duration", "pupil_r", "pupil_l", "trial"]]
        # We filtered uncorrect fixations placed outside the screen
        fixations = fixations[(fixations['y'] <= 1)  & (fixations['y'] >= 0) & (fixations['x'] <= 1) &  (fixations['x'] >= 0)]
        return fixations

    def filter_files_byseconds(self, fixations: Optional[pd.DataFrame], gaze: Optional[pd.DataFrame], max_seconds: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        fix = None
        gaz = None
        
        if fixations is not None:
            fix = copy.deepcopy(fixations)
            fix = fix.sort_values(by='TIME')
            fix["TIME"] = fix["TIME"].astype(float)
            start = fix.iloc[0]['TIME']
            fix['TIME'] = fix['TIME'] - start
            fix = fix[fix['TIME'] <= max_seconds]
            
            if gaze is not None:
                gaz = copy.deepcopy(gaze)
                gaz["TIME"] = gaz["TIME"].astype(float)
                gaz['TIME'] = gaz['TIME'] - start
                gaz = gaz[gaz['TIME'] <= max_seconds]
                
        elif gaze is not None:
            gaz = copy.deepcopy(gaze)
            gaz = gaz.sort_values(by='TIME') 
            gaz["TIME"] = gaz["TIME"].astype(float)
            start = gaz.iloc[0]['TIME']
            gaz['TIME'] = gaz['TIME'] - start
            gaz = gaz[gaz['TIME'] <= max_seconds]
            
        return fix, gaz

    def compute_reading_measures(self, fixations:Optional[pd.DataFrame], gaze:Optional[pd.DataFrame]) -> Dict:
        if fixations is None and gaze is None:
            raise ValueError("At least one of fixations or gaze must be provided")
        reading_measures = {}
        if fixations is not None:
            reading_measures['fix_number'] = len(fixations)
            reading_measures['fix_duration'] = fixations['duration'].sum()
        if gaze is not None:
            reading_measures['pupil'] = gaze['pupil'].astype(float).mean()
        return reading_measures
    
    def filter_fixations_trial(self, trial:str, fixations:pd.DataFrame) -> pd.DataFrame:
        return fixations[fixations['trial'] == trial]
    
    def filter_gaze_trial(self, trial:str, gaze:pd.DataFrame) -> pd.DataFrame:    
        return gaze[gaze['trial'] == trial]
    
    @staticmethod
    def _scale_column(df:pd.DataFrame, column:str) -> pd.DataFrame:
        scaler = MinMaxScaler()
        df[[column]] = scaler.fit_transform(df[[column]])
        return df

    def preprocess_gaze(self, df:pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess_fixations(df)
        df['pupil'] = (df['pupil_r'] + df['pupil_l']) / 2
        df.drop(columns=['pupil_r', 'pupil_l'], inplace=True)
        df = self._scale_column(df, 'pupil')
        return df



        

