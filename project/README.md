# Arguments

`> python main.py <AGENT> [..Options]`  
`string Agent: ['agent', 'rule'], default: 'agent'`

## Agent
### Run (Default)
```text
Arguments:
Options:
	-d --dry			BOOL	Whether to run as dry and export a gif
	-p --port			INT		The port to run the webserver on (default: 8000)
```
### Train
```text
Arguments:
Options:
	-s --steps		INT		How many steps to train for (default: 20000)
```

## Rule based
```text
Arguments:
Options:
	-d --dry			BOOL	Whether to run as dry and export a gif
	-i --iterations	    INT		Iterations? (default: 64)
	-p --port			INT		The port to run the webserver on (default: 8000)
```


# TODO