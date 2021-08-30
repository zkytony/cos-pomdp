# thor


## Caveats

- Navigation actions for COSPOMDP are
  defined in `cospomdp_apps.basic.action`;
  Only the action name is used to find the corresponding
  ai2thor action. Thus it is on you to make sure
  the parameters of the COSPOMDP actions match up
  with the parameters of the ai2thor actions.

  (This should be fixed)

   08/30/21: This is fixed.

- Should always use `thortils.get_navigation_actions` and `thortils.convert_movement_to_actions`
  when converting from movement with parameters defined in thor coordinate system (i.e. movement_params)
  to a movement tuple or a POMDP action for the movement.

  - `ThorObjectSearchOptimalAgent` uses this by passing movement_params to
    the get_shortest_path_to_object function, that is part of 'nav_config'.

  - `ThorObjectSearchBasicCosAgent` uses this by converting the output POMDP
    action to a (name, params) tuple, where params are in ai2thor units.

  - `ThorObjectSearchExternalAgent` does not do this because its model's output
    is outside of our control.
