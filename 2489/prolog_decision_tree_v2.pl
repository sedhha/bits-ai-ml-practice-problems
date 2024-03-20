% Define the main entry point
start :-
    writeln('Enter the lake distance in Kms: '),
    read(LakeDistance),
    (   LakeDistance < 10
    ->  writeln('Use lake water.')
    ;   LakeDistance >= 10
    ->  river_distance
    ).

% Rules for River Distance
river_distance :- 
    writeln('Enter the river distance in Kms: '),
    read(RiverDistance),
    (   RiverDistance >= 8
    ->  high_river_distance
    ;   RiverDistance < 8
    ->  low_river_distance
    ).

% Rules for Heavy Rain Intensity
high_river_distance :-
    writeln('Enter the monthly rainfall in mm: '),
    read(Rainfall),
    (   Rainfall < 150
    ->  sandy_aquifier
    ;   Rainfall >= 150
    ->  writeln('Use rain water.')
    ).

% Rules for Low Rain Intensity
low_river_distance :-
    writeln('Enter the monthly rainfall in mm: '),
    read(Rainfall),
    (   Rainfall < 200
    ->  writeln('Use river water.')
    ;   Rainfall >= 200
    ->  writeln('Use rain water.')
    ).

% Rules for Sandy Aquifier
sandy_aquifier :-
    writeln('Is there a sandy aquifier? (yes/no)?: '),
    read(Aquifier),
    (   Aquifier == no
    ->  lake_distance
    ;   Aquifier == yes
    ->  beach_distance        
    ).

% Rules for Lake Distance
lake_distance :-
    writeln('Enter the lake distance in Kms: '),
    read(LakeDistance),
    (   LakeDistance < 14
    ->  writeln('Use lake water.')
    ;   LakeDistance >= 14
    ->  writeln('Use rain water.')
    ).

% Rules for Beach Distance
beach_distance :-
    writeln('Enter the beach distance in Kms: '),
    read(BeachDistance),
    (   BeachDistance < 5
    ->  river_distance_v2
    ;   BeachDistance >= 5
    ->  writeln('Use ground water.')
    ).

% Rules for River Distance V2
river_distance_v2 :-
    writeln('Enter the river distance in Kms: '),
    read(RiverDistance),
    (   RiverDistance < 20
    ->  writeln('Use river water.')
    ;   RiverDistance >= 20
    ->  writeln('Use rain water.')
    ).
