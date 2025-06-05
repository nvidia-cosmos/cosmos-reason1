## Cosmos Command Line Tool


After installation of cosmos_reason1, there will have a `cosmos` command added to the system. This CLI will help user to query information from controller. For each command, user should pass the controller address and port to it.


```
cosmos --help

Usage: cosmos [OPTIONS] COMMAND [ARGS]...

  Cosmos Reason1 CLI.

Options:
  --help  Show this message and exit.

Commands:
  algo      Query information about training config from controller.
  nccl      Query information about NCCL from controller.
  profile   Manage profiler behavior of the training.
  replica   Query replica information from controller.
```

### List replicas


```
cosmos replica ls -ch localhost -cp 8000
```


Output:

```
                                                 Replicas                                                  
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Replica                              ┃ Role    ┃ Arrived ┃ Atom Number ┃ Weight Step ┃ Pending Rollouts ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ 7b987ce5-18bf-4499-9ed0-04f689ab91d7 │ POLICY  │ Yes     │ 2           │ 0           │ 8                │
├──────────────────────────────────────┼─────────┼─────────┼─────────────┼─────────────┼──────────────────┤
│ 877e7132-6156-4ac1-b1ca-8a5b729c9d82 │ ROLLOUT │ Yes     │ 2           │ 0           │ N/A              │
└──────────────────────────────────────┴─────────┴─────────┴─────────────┴─────────────┴──────────────────┘
```

If you want check details of a replica:

```
cosmos replica lsr -ch localhost -cp 8000 7b987ce5-18bf-4499-9ed0-04f689ab91d7
```

Output:

```

      Replica : 7b987ce5-18bf-4499-9ed0-04f689ab91d7, Role: POLICY       
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Atom Global Rank ┃ Host IP     ┃ Host Name               ┃ Trace Path ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 1                │ 10.65.29.83 │ cw-dfw-h100-004-308-012 │ None       │
├──────────────────┼─────────────┼─────────────────────────┼────────────┤
│ 0                │ 10.65.29.83 │ cw-dfw-h100-004-308-012 │ None       │
└──────────────────┴─────────────┴─────────────────────────┴────────────┘

```


### List config

```
cosmos algo config -ch localhost -cp 8000
```

This will output the config of cosmos.

### List NCCL info

```
cosmos nccl ls -ch localhost -cp 8000
```

This will list the NCCL info stored in controller.

### Interactive with profiler

Please follow the [Profiler](docs/Profiler.md)
