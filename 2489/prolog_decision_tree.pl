% Define the main entry point
start :-
    prompt_lake_distance,
    prompt_river_distance,
    prompt_rainfall,
    prompt_aquifer,
    prompt_beach_distance,
    make_decision.

% Prompt for the lake distance
prompt_lake_distance :-
    writeln('Enter the lake distance in Kms: '),
    read(LakeDistance),
    asserta(lake_distance(LakeDistance)).

% Prompt for the river distance
prompt_river_distance :-
    writeln('Enter the river distance in Kms: '),
    read(RiverDistance),
    asserta(river_distance(RiverDistance)).

% Prompt for the monthly rainfall
prompt_rainfall :-
    writeln('Enter the monthly rainfall in mm: '),
    read(Rainfall),
    asserta(rainfall(Rainfall)).

% Prompt for the presence of a sandy aquifer
prompt_aquifer :-
    writeln('Is there a sandy aquifer? (yes/no)?: '),
    read(Aquifer),
    asserta(aquifer(Aquifer)).

% Prompt for the beach distance
prompt_beach_distance :-
    writeln('Enter the beach distance in Kms: '),
    read(BeachDistance),
    asserta(beach_distance(BeachDistance)).

% Make the decision based on the inputs
make_decision :-
    lake_distance(LakeDistance),
    (   LakeDistance < 10
    ->  writeln('Use lake water.')
    ;   river_distance(RiverDistance),
        (   RiverDistance >= 8
        ->  rainfall(Rainfall),
            (   Rainfall < 150
            ->  aquifer(Aquifer),
                (   Aquifer == yes
                ->  beach_distance(BeachDistance),
                    (   BeachDistance < 5
                    ->  (   RiverDistance < 20
                        ->  writeln('Use river water.')
                        ;   writeln('Use rain water.')
                        )
                    ;   writeln('Use ground water.')
                    )
                ;   LakeDistance < 14
                ->  writeln('Use lake water.')
                ;   writeln('Use rain water.')
                )
            ;   writeln('Use rain water.')
            )
        ;   rainfall(Rainfall),
            (   Rainfall < 200
            ->  writeln('Use river water.')
            ;   writeln('Use rain water.')
            )
        )
    ).

% Clean up the asserted facts to avoid side effects in subsequent runs
cleanup :-
    retractall(lake_distance(_)),
    retractall(river_distance(_)),
    retractall(rainfall(_)),
    retractall(aquifer(_)),
    retractall(beach_distance(_)).

% After decision, run the cleanup
:- start, cleanup.
