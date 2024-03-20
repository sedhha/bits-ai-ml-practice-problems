% Define the main entry point
start :- 
    writeln('Enter the monthly rainfall in mm:'),
    read(Rainfall),
    (
        Rainfall < 150,
        -> sandy_aquifier(Rainfall)
        ; Rainfall >= 150,
          Rainfall < 200
        -> writeln('Use rainwater.')
        ;
            Rainfall >= 200
        -> writeln('Use rainwater.')
    ).

% Rules for Sandy Aquifier

sandy_aquifier(Rainfall) :-
    writeln('Is there a sandy aquifier? (yes/no)?: '),
    read(Aquifier),
    (
        Aquifier == yes
        -> beach_distance(Rainfall)
        ; Aquifier == no
        -> writeln('Use lake water.')
    ).

% Rules for Beach Distance

beach_distance(Rainfall) :-
    writelen('Enter the beach distance in Kms: '),
    read(BeachDistance),
    (
        BeachDistance < 5
        -> river_distance(Rainfall)
        ; BeachDistance >= 5
        -> writeln('Use ground water')
    ).

% Rules for River Distance 

river_distance(Rainfall) :-
    writeln('Enter the river distance in Kms: '),
    read(RiverDistance)
    (
        RiverDistance < 20
        -> writeln('Use river water.')
        ;
        RiverDistance >= 20
        -> writeln('Use rain water.')
    )