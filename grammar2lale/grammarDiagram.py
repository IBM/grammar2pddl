from railroad import NonTerminal, Terminal, Choice, OneOrMore, ZeroOrMore, Diagram, Optional, Sequence, Group, MultipleChoice, Comment, Start, DEFAULT_STYLE

tfm_diagram = Group(Choice(0, "Normalizer", "MinMaxScaler", "RobustScaler", "StandardScaler", "PCA"), "tfm")
est_diagram = Group(Choice(0, "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier","GaussianNB", "KNeighbors", "QDA"),"est")
sklearn_diagram = Diagram("sklearn_subset", Sequence(
    Group(
        Choice(0,
            tfm_diagram,
            Sequence(tfm_diagram, NonTerminal("DAG")),
            Sequence(MultipleChoice(0, "all", tfm_diagram, NonTerminal("DAG")), "Concat" ),
            Sequence(MultipleChoice(0, "all", est_diagram, NonTerminal("DAG")), "Concat" ),
        ),
        "DAG"
    ),
    Group(
        est_diagram, 
        "est"
    )
))