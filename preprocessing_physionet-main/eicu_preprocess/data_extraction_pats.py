# eicu_preprocess/data_extraction_pats.py
import os
import pickle
from utils.utils import dataframe_from_csv
from utils.pat_utils import (
    filter_patients_on_age,
    filter_one_unit_stay,
    filter_patients_on_columns,
    transform_gender,
    transform_ethnicity,
    transform_hospital_discharge_status,
    transform_unit_discharge_status,
    transform_dx_into_id,
    filter_patients_on_columns_model,
    filter_max_hours,
    create_labels
)

# ⚠️ MODIFIÉ : Chemin vers eICU-CRD Demo
eicu_path = '../../data/real/eicu/raw/eicu-collaborative-research-database-demo-2.0.1'

print("=" * 70)
print("===========> Processing patient chart <===========")
print("=" * 70)


def read_patients_table(eicu_path):
    """Charger et filtrer la table des patients"""

    # 1. Charger
    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv.gz'), index_col=False)
    print(f"\n📊 Patients initiaux : {len(pats):,}")

    # 2. Filtrer par âge
    pats = filter_patients_on_age(pats, min_age=15, max_age=89)

    # 3. Un séjour par patient
    pats = filter_one_unit_stay(pats)

    # 4. Durée minimale de séjour
    pats = filter_max_hours(pats, max_hours=24, thres=240)

    # 5. Colonnes critiques
    pats = filter_patients_on_columns(pats)

    # 6. Transformations
    print("\n🔄 Transformations...")
    pats['gender'] = transform_gender(pats.gender)
    pats['ethnicity'] = transform_ethnicity(pats.ethnicity)
    pats['hospitaldischargestatus'] = transform_hospital_discharge_status(
        pats.hospitaldischargestatus
    )
    pats['unitdischargestatus'] = transform_unit_discharge_status(
        pats.unitdischargestatus
    )

    # 7. Diagnostics
    pats = transform_dx_into_id(pats)

    # 8. Sélection colonnes
    pats = filter_patients_on_columns_model(pats)

    # 9. Créer labels
    pats = create_labels(pats)

    # 10. Cohorte finale
    cohort = pats.patientunitstayid.unique()

    print("\n" + "=" * 70)
    print(f"✅ COHORTE FINALE : {len(cohort):,} patients uniques")
    print("=" * 70)

    return pats, cohort


if __name__ == '__main__':
    # Exécuter le preprocessing
    pats, cohort = read_patients_table(eicu_path)

    # Sauvegarder
    print("\n💾 Sauvegarde des résultats...")

    output_dir = 'output/patient'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'pats.pkl'), 'wb') as outfile:
        pickle.dump(pats, outfile, pickle.HIGHEST_PROTOCOL)
    print(f"  ✅ {output_dir}/pats.pkl")

    with open(os.path.join(output_dir, 'cohort.pkl'), 'wb') as outfile:
        pickle.dump(cohort, outfile, pickle.HIGHEST_PROTOCOL)
    print(f"  ✅ {output_dir}/cohort.pkl")

    print("\n" + "=" * 70)
    print("🎉 Preprocessing terminé avec succès !")
    print("=" * 70)






# import os
# import pickle
# from utils.utils import dataframe_from_csv
# from utils.pat_utils import filter_patients_on_age, filter_one_unit_stay, filter_patients_on_columns, \
#     transform_gender, transform_ethnicity, transform_hospital_discharge_status, transform_unit_discharge_status, \
#     transform_dx_into_id, filter_patients_on_columns_model, filter_max_hours, create_labels
#
# eicu_path = '../../../eICU_data/eicu-collaborative-research-database-2.0'
#
# print("===========> Processing patient chart <===========")
# def read_patients_table(eicu_path):
#     pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv'), index_col=False)
#     pats = filter_patients_on_age(pats, min_age=15, max_age=89)
#     pats = filter_one_unit_stay(pats)
#     pats = filter_max_hours(pats, max_hours=24, thres=240)
#     pats = filter_patients_on_columns(pats)
#
#     pats.update(transform_gender(pats.gender))
#     pats.update(transform_ethnicity(pats.ethnicity))
#     pats.update(transform_hospital_discharge_status(pats.hospitaldischargestatus))
#     pats.update(transform_unit_discharge_status(pats.unitdischargestatus))
#     pats = transform_dx_into_id(pats)
#     pats = filter_patients_on_columns_model(pats)
#     pats = create_labels(pats)
#
#     cohort = pats.patientunitstayid.unique()
#     print("number of the cohort (unique patients): ", len(cohort))
#     return pats, cohort
# pats, cohort = read_patients_table(eicu_path)
#
# with open('output/patient/pats.pkl', 'wb') as outfile:
#     pickle.dump(pats, outfile, pickle.HIGHEST_PROTOCOL)
# with open('output/patient/cohort.pkl', 'wb') as outfile:
#     pickle.dump(cohort, outfile, pickle.HIGHEST_PROTOCOL)