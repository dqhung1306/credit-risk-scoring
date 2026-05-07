import numpy as np
import pandas as pd
import gc
import re
class DataPreprocessor:
    def __init__(self, is_training=True):
        self.is_training = is_training
        # Biến này để lưu lại bộ cột gốc, dùng cho Web App
        self.train_features = None 
    def _create_logs(self, data, features):
        for var in features:
            if var in data.columns:
                data["LOG_" + str(var)] = np.log(data[var].abs() + 1)
                data.drop(columns=str(var), inplace=True)
        return data

    def processing_application(self, data):
        print("Processing Application...")
        application = data.copy()    
        
        # INCOME RATIO - safe division with check
        if all(col in application.columns for col in ["AMT_CREDIT", "AMT_INCOME_TOTAL"]):
            application["CREDIT_BY_INCOME"] = application["AMT_CREDIT"] / application["AMT_INCOME_TOTAL"]
        if all(col in application.columns for col in ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]):
            application["ANNUITY_BY_INCOME"] = application["AMT_ANNUITY"] / application["AMT_INCOME_TOTAL"]
        if all(col in application.columns for col in ["AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]):
            application["GOODS_PRICE_BY_INCOME"] = application["AMT_GOODS_PRICE"] / application["AMT_INCOME_TOTAL"]
        if all(col in application.columns for col in ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"]):
            application["INCOME_PER_PERSON"] = application["AMT_INCOME_TOTAL"] / application["CNT_FAM_MEMBERS"]
        
        # CREDIT 
        if all(col in application.columns for col in ["AMT_CREDIT", "AMT_GOODS_PRICE"]):
            application["CREDIT_TO_GOODS_RATIO"] = application["AMT_CREDIT"] / application["AMT_GOODS_PRICE"]
        if all(col in application.columns for col in ["AMT_CREDIT", "AMT_ANNUITY"]):
            application['ANNUITY LENGTH'] = application['AMT_CREDIT'] / application['AMT_ANNUITY']
        
        # FAMILY STATUS
        if all(col in application.columns for col in ["CNT_FAM_MEMBERS", "CNT_CHILDREN"]):
            application["CNT_ADULTS"] = application["CNT_FAM_MEMBERS"] - application["CNT_CHILDREN"]
            application['CHILDREN_RATIO'] = application['CNT_CHILDREN'] / application['CNT_FAM_MEMBERS']
        
        # EMPLOYMENT RATE
        if 'DAYS_EMPLOYED' in application.columns:
            application['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        if all(col in application.columns for col in ["DAYS_EMPLOYED", "DAYS_BIRTH"]):
            application["EMPLOYED_TO_BIRTH_RATIO"] = application["DAYS_EMPLOYED"] / application["DAYS_BIRTH"]
        
        # NUMBER OF DOCUMENTS
        doc_vars = ["FLAG_DOCUMENT_2",  "FLAG_DOCUMENT_3",  "FLAG_DOCUMENT_4",  "FLAG_DOCUMENT_5",  "FLAG_DOCUMENT_6",
                    "FLAG_DOCUMENT_7",  "FLAG_DOCUMENT_8",  "FLAG_DOCUMENT_9",  "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11",
                    "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
                    "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"]
        doc_cols_exist = [col for col in doc_vars if col in application.columns]
        if doc_cols_exist:
            application["NUM_DOCUMENTS"] = application[doc_cols_exist].sum(axis=1)
        application.drop(columns=doc_vars, inplace=True, errors='ignore')
        
        if all(col in application.columns for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
            application["EXT_SOURCE_MEAN"] = application[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
        
        # CONTACT
        contact_vars = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
        contact_cols_exist = [col for col in contact_vars if col in application.columns]
        if contact_cols_exist:
            application["NUM_CONTACT_METHODS"] = application[contact_cols_exist].sum(axis=1)
        application.drop(columns=contact_vars, inplace=True, errors='ignore')
        
        # CAR
        if all(col in application.columns for col in ["OWN_CAR_AGE", "DAYS_BIRTH"]):
            application["CAR_AGE_TO_BIRTH_RATIO"] = application["OWN_CAR_AGE"] / (application["DAYS_BIRTH"] / -365.25)

        # CONVERT DAYS
        day_vars = ["DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"]
        if "DAYS_BIRTH" in application.columns:
            application["AGE"] = np.round(application["DAYS_BIRTH"] / (-365), 0)
        if "DAYS_EMPLOYED" in application.columns:
            application["YEARS_EMPLOYED"] = np.round(application["DAYS_EMPLOYED"] / (-365), 0)
        if "DAYS_REGISTRATION" in application.columns:
            application["YEARS_REGISTRATION"] = np.round(application["DAYS_REGISTRATION"] / (-365), 0)
        if "DAYS_ID_PUBLISH" in application.columns:
            application["YEARS_ID_PUBLISH"] = np.round(application["DAYS_ID_PUBLISH"] / (-365), 0)
        if "DAYS_LAST_PHONE_CHANGE" in application.columns:
            application["YEARS_LAST_PHONE_CHANGE"] = np.round(application["DAYS_LAST_PHONE_CHANGE"] / (-365), 0)
        
        # Cleanup negative values
        for col in ["AGE", "YEARS_EMPLOYED", "YEARS_REGISTRATION", "YEARS_ID_PUBLISH", "YEARS_LAST_PHONE_CHANGE"]:
            if col in application.columns:
                application.loc[application[col] < 0, col] = np.nan

        application.drop(columns=day_vars, inplace=True, errors='ignore')
        
        # Gọi hàm _create_logs nội bộ
        log_vars = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "AMT_ANNUITY"]
        application = self._create_logs(application, log_vars)
        
        # COLUMNS TO DROP
        drops = ['APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 
                 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
                 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
                 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'COMMONAREA_MODE','ELEVATORS_MODE', 'ENTRANCES_MODE', 
                 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 
                 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'TOTALAREA_MODE',  'YEARS_BEGINEXPLUATATION_MODE', 
                 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'FONDKAPREMONT_MODE']
        application = application.drop(columns=drops, errors='ignore')
        
        # Handle ORGANIZATION_TYPE if it exists
        if 'ORGANIZATION_TYPE' in application.columns:
            def group_organization(org_name):
                if pd.isna(org_name):
                    return 'XNA'
                if 'Business Entity' in org_name:
                    return 'Business_Entity'
                elif 'Industry' in org_name:
                    return 'Industry'
                elif 'Trade' in org_name:
                    return 'Trade'
                elif 'Transport' in org_name:
                    return 'Transport'
                elif org_name in ['XNA', 'Self-employed', 'Medicine', 'Government', 'School', 'Construction', 'Kindergarten']:
                    return org_name
                else:
                    return 'Other'
            application['ORGANIZATION_TYPE_GROUPED'] = application['ORGANIZATION_TYPE'].apply(group_organization)
            application.drop(columns=['ORGANIZATION_TYPE'], inplace=True, errors='ignore')
        return application

    def processing_bureau(self, application, bureau, bureau_balance):
        print(70*"=")
        print("Processing Bureau...")
        bbal                = bureau_balance.copy()
        bure                = bureau.copy()
        status_map          = {'C': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'X': np.nan}
        bbal["NUM_STATUS"]  = bbal["STATUS"].map(status_map)
        bbal["LOAN_SCORE"]  = bbal["NUM_STATUS"] / (abs(bbal["MONTHS_BALANCE"]) + 1)
        loan_score          = bbal.groupby("SK_ID_BUREAU", as_index = False).LOAN_SCORE.sum()
        del bbal["NUM_STATUS"], bbal["LOAN_SCORE"]

        bbal        = pd.get_dummies(bbal, columns = ["STATUS"], prefix = "STATUS")
        cnt_mon     = bbal.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].count()
        del bbal["MONTHS_BALANCE"]
        agg_bbal    = bbal.groupby("SK_ID_BUREAU").mean()
        agg_bbal["MONTH_COUNT"] = cnt_mon
        agg_bbal    = agg_bbal.reset_index()
        agg_bbal    = agg_bbal.merge(loan_score, how = "left", on = "SK_ID_BUREAU")

        if 'SK_ID_BUREAU' not in agg_bbal.columns:
            agg_bbal = agg_bbal.reset_index()
            
        bure = bure.merge(agg_bbal, how='left', on='SK_ID_BUREAU')
        bure = pd.get_dummies(bure, columns=['CREDIT_ACTIVE'], prefix='BUREAU_STATUS')

        balance_cols = [c for c in bure.columns if c.startswith('STATUS_')] + ['LOAN_SCORE', 'MONTH_COUNT']
        
        # Build agg_dict dynamically to handle missing categories
        agg_dict = {
            'AMT_CREDIT_SUM_DEBT'   : 'sum',      
            'AMT_CREDIT_SUM'        : 'sum',            
            'AMT_ANNUITY'           : 'sum',               
            'AMT_CREDIT_SUM_OVERDUE': 'sum',   
            'AMT_CREDIT_MAX_OVERDUE': 'max',    
            'CREDIT_DAY_OVERDUE'    : 'max',      
            'CNT_CREDIT_PROLONG'    : 'sum'         
        }
        
        # Add BUREAU_STATUS columns dynamically (only if they exist)
        for col in bure.columns:
            if col.startswith('BUREAU_STATUS_'):
                agg_dict[col] = 'sum'
        
        # Add balance columns dynamically
        for col in balance_cols:
            if col in bure.columns and col not in agg_dict:
                agg_dict[col] = 'mean'

        valid_agg_dict = {k: v for k, v in agg_dict.items() if k in bure.columns}
        bureau_agg = bure.groupby('SK_ID_CURR').agg(valid_agg_dict).reset_index()
        
        bureau_agg['BUREAU_CREDIT_UTILIZATION'] = bureau_agg['AMT_CREDIT_SUM_DEBT'] / bureau_agg['AMT_CREDIT_SUM']
        bureau_agg['BUREAU_CREDIT_UTILIZATION'].replace([np.inf, -np.inf], 0, inplace=True)

        rename_dict = {
            'AMT_CREDIT_SUM_DEBT': 'BUREAU_TOTAL_DEBT',
            'AMT_CREDIT_SUM': 'BUREAU_TOTAL_LIMIT',
            'AMT_ANNUITY': 'BUREAU_TOTAL_ANNUITY',
            'AMT_CREDIT_SUM_OVERDUE': 'BUREAU_CURRENT_OVERDUE_AMT',
            'AMT_CREDIT_MAX_OVERDUE': 'BUREAU_MAX_OVERDUE_AMT',
            'CREDIT_DAY_OVERDUE': 'BUREAU_MAX_DAYS_OVERDUE',
            'CNT_CREDIT_PROLONG': 'BUREAU_TOTAL_PROLONG_CNT'
        }
        bureau_agg.rename(columns=rename_dict, inplace=True)

        application = application.merge(bureau_agg, how='left', on='SK_ID_CURR')
        
        bureau_cols = [c for c in bureau_agg.columns if c != 'SK_ID_CURR']
        application[bureau_cols] = application[bureau_cols].fillna(0)
        
        return application
    
    def processing_prev_application(self, prev_application):
        print(70*"=")
        print("Processing Previous Application...")
        df = prev_application.copy()
        df['PREV_CREDIT_GAP_RATIO'] = df['AMT_CREDIT'] / df['AMT_APPLICATION']
        df['PREV_CREDIT_GAP_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df = pd.get_dummies(df, columns=['NAME_CONTRACT_STATUS'], prefix='PREV_STATUS')
        
        # Build agg_dict dynamically to handle missing categorical columns
        agg_dict = {
            'SK_ID_PREV': 'count',                 
            'PREV_CREDIT_GAP_RATIO': ['mean', 'min'], 
            'DAYS_DECISION': 'max',                
            'RATE_DOWN_PAYMENT': ['mean', 'max']   
        }
        
        # Add PREV_STATUS columns only if they exist
        for col in df.columns:
            if col.startswith('PREV_STATUS_'):
                agg_dict[col] = 'mean'
        
        prev_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        return prev_agg.reset_index()

    def processing_installments(self, inst):
        print(70*"=")
        print("Processing Installments Payments...")
        df = inst.copy()
        df['DPD']       = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
        df['IS_LATE']   = (df['DPD'] > 0).astype(int)
        df['IS_EARLY']  = (df['DPD'] <= 0).astype(int)
        df['DPD_ONLY']  = df['DPD'].apply(lambda x: x if x > 0 else 0)
        df['PAYMENT_DEFICIT_RATIO'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
        df['PAYMENT_DEFICIT_RATIO'].replace([np.inf, -np.inf], 1, inplace=True)
        
        agg_dict = {
            'SK_ID_PREV': 'count',               
            'DPD_ONLY': ['max', 'mean', 'sum'],  
            'IS_LATE': 'sum',                    
            'IS_EARLY': 'sum',                   
            'PAYMENT_DEFICIT_RATIO': 'mean'     
        }
        
        # Add NUM_INSTALMENT_VERSION only if it exists
        if 'NUM_INSTALMENT_VERSION' in df.columns:
            agg_dict['NUM_INSTALMENT_VERSION'] = 'nunique'
        
        inst_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
        inst_agg.columns = pd.Index(['INST_' + e[0] + "_" + e[1].upper() for e in inst_agg.columns.tolist()])
        return inst_agg.reset_index()

    def processing_credit_card(self, cc):
        print(70*"=")    
        print("Processing Credit Card Balance")
        df = cc.copy()
        
        # Calculate ratios safely
        df['CC_UTILIZATION']    = df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL']
        df['CC_CASH_OUT_RATIO'] = df['AMT_DRAWINGS_ATM_CURRENT'] / df['AMT_DRAWINGS_CURRENT']
        df['CC_MIN_PAY_STRESS'] = df['AMT_PAYMENT_CURRENT'] / df['AMT_INST_MIN_REGULARITY']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        agg_dict = {
            'MONTHS_BALANCE': 'count',
            'CC_UTILIZATION': ['max', 'mean'],
            'CC_CASH_OUT_RATIO': ['max', 'mean'],
            'CC_MIN_PAY_STRESS': 'mean'
        }
        
        # Add SK_DPD only if it exists
        if 'SK_DPD' in df.columns:
            agg_dict['SK_DPD'] = 'max'
        
        # Only keep columns that exist in the dataframe
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        cc_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
        return cc_agg.reset_index()

    def processing_pos_cash(self, pos):
        print(70*"=")
        print("Processing POS_CASH_balance")
        df = pos.copy()
        df['POS_LOAN_PROGRESS'] = df['CNT_INSTALMENT_FUTURE'] / df['CNT_INSTALMENT']
        df['POS_LOAN_PROGRESS'].replace([np.inf, -np.inf], 0, inplace=True)
        df = pd.get_dummies(df, columns=['NAME_CONTRACT_STATUS'], prefix='POS_STATUS')

        agg_dict = {
            'MONTHS_BALANCE': 'count',
            'POS_LOAN_PROGRESS': 'mean',
            'SK_DPD': 'max',
            'SK_DPD_DEF': 'max',
            **{col: 'mean' for col in df.columns if col.startswith('POS_STATUS_')}
        }
        pos_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
        new_cols = []
        for col in pos_agg.columns:
            if col.startswith('POS_') or col.startswith('SK_'):
                new_cols.append(col)
            else:
                new_cols.append('POS_' + col) 
        pos_agg.columns = new_cols
        return pos_agg.reset_index()
    def datapreprocessing(self, app_data, bureau=None, bureau_balance=None, 
                          prev_app=None, inst=None, pos=None, cc=None):
        df = self.processing_application(app_data)
        if bureau is not None and bureau_balance is not None:
            df = self.processing_bureau(df, bureau, bureau_balance)
            gc.collect()

        if prev_app is not None:
            prev_agg = self.processing_prev_application(prev_app)
            df = df.merge(prev_agg, how='left', on='SK_ID_CURR')
            del prev_agg; gc.collect()

        if inst is not None:
            inst_agg = self.processing_installments(inst)
            df = df.merge(inst_agg, how='left', on='SK_ID_CURR')
            del inst_agg; gc.collect()

        if pos is not None:
            pos_agg = self.processing_pos_cash(pos)
            df = df.merge(pos_agg, how='left', on='SK_ID_CURR')
            del pos_agg; gc.collect()

        if cc is not None:
            cc_agg = self.processing_credit_card(cc)
            df = df.merge(cc_agg, how='left', on='SK_ID_CURR')
            del cc_agg; gc.collect()
        print(70*"=")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        count_sum_cols      = [col for col in df.columns if 'AMT_' in col or 'NUM_' in col or '_COUNT' in col or '_TOTAL' in col or '_SUM' in col]
        df[count_sum_cols]  = df[count_sum_cols].fillna(0)
        categorical_cols    = df.select_dtypes(include=['object', 'category']).columns
        df                  = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df                  = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', str(x)))

        if self.is_training:
            self.train_features = [c for c in df.columns if c != 'TARGET']
        else:
            if self.train_features is None:
                raise ValueError("Train first")
            
            missing_cols = set(self.train_features) - set(df.columns)
            for c in missing_cols:
                df[c] = np.nan
                
            extra_cols = set(df.columns) - set(self.train_features) - {'SK_ID_CURR'}
            df = df.drop(columns=list(extra_cols), errors='ignore')
            
            cols_order = ['SK_ID_CURR'] + [c for c in self.train_features if c != 'SK_ID_CURR']
            df = df[cols_order]

        print(f"Done - New data shape: {df.shape}")
        return df