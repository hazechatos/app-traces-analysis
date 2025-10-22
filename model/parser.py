import re

def parse_action_string(action_string):
    """
    Parse une chaîne d'action en un tuple (template, c1, c2, c3).
    
    Pattern attendu: template(c1)<c2>$c3$
    - template: partie principale de l'action
    - c1: contenu entre parenthèses () (optionnel)
    - c2: contenu entre chevrons <> (optionnel)
    - c3: contenu après le dernier $ (optionnel)
    
    Args:
        action_string (str): La chaîne d'action à parser
        
    Returns:
        tuple: (template, c1, c2, c3) où chaque élément peut être None si absent
    """
    if not action_string:
        return (None, None, None, None)
    
    # Initialiser les variables
    template = action_string
    c1 = None
    c2 = None
    c3 = None
    
    # Extraire c3 (contenu après le dernier $)
    if '$' in action_string:
        parts = action_string.split('$')
        if len(parts) > 1:
            c3 = parts[-1] if parts[-1] else None
            # Reconstruire la chaîne sans la partie c3
            action_string = '$'.join(parts[:-1])
    
    # Extraire c2 (contenu entre chevrons <>)
    c2_match = re.search(r'<([^>]*)>', action_string)
    if c2_match:
        c2 = c2_match.group(1) if c2_match.group(1) else None
        # Retirer la partie c2 de la chaîne
        action_string = re.sub(r'<[^>]*>', '', action_string)
    
    # Extraire c1 (contenu entre parenthèses ())
    c1_match = re.search(r'\(([^)]*)\)', action_string)
    if c1_match:
        c1 = c1_match.group(1) if c1_match.group(1) else None
        # Retirer la partie c1 de la chaîne
        action_string = re.sub(r'\([^)]*\)', '', action_string)
    
    # Le template est ce qui reste après avoir extrait c1, c2, c3
    template = action_string.strip()
    
    return (template, c1, c2, c3)

# Tests avec les exemples fournis
def test_parse_action():
    """Test de la fonction parse_action_string avec les exemples fournis"""
    
    test_cases = [
        "Fermeture d'un panel(MAINT)<DEF_03/24>$VT$",
        "Sélection d'un écran<DEF_03/24>$GESCO$",
        "Retour sur un écran(infologic.acti.modules.AT_ACTIVITES.tache.dialog.ActiTacheController)<blocagePourRetard>",
        "Clic sur une grille d'historique de recherche(PATCHPROD)",
        "Retour sur un écran(infologic.orga.modules.OR_PROJET.ProjetORForm)<Projet_DEV>1",
        "Saisie dans un champ(infologic.acti.modules.AT_ACTIVITES.ficheactivite.ActiFicheActiviteController)<STD>1",
        "Retour sur un écran(DOC)$XPL$",
        "Saisie dans un champ$XPL$1",
        "Exécution d'un bouton$XPL$1",
        "Affichage d'une erreur$XPL$1",
        "Fermeture d'une dialogue$XPL$1",
        "Saisie dans un champ(infologic.core.gui.controllers.BlankController)<PORTAGE_WEB>",
        "Création d'un écran(infologic.core.accueil.AccueilController)t5",
        "Sélection d'un écran(TESTOBS)<LERCNAT>",
        "Chainage(MAINT)<DEF_03/24>$ST$",
        "Chainage(BUG)$MES$",
        "Fermeture de session(VT_EditDevis)",
        "Chainage(DEV)$INFO$"
    ]
    
    print("Tests de la fonction parse_action_string:")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        result = parse_action_string(test_case)
        print(f"Test {i:2d}: {test_case}")
        print(f"        Résultat: {result}")
        print()

if __name__ == "__main__":
    test_parse_action()