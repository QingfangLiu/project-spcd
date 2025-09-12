### Uploading SPCD data to OpenNeuro

#### OpenNeuro Command Line Interface

  * **Documentation:** [https://docs.openneuro.org/packages/openneuro-cli.html](https://docs.openneuro.org/packages/openneuro-cli.html)

#### OpenNeuro Accession Number

  * `ds006081`

-----

#### Pre-flight checks and data preparation

To test if Deno has been installed and is ready to run, use the following command:

```
deno --version
```

To remove all redundant files from the dataset folder before uploading:

```
find . -name ".DS_Store" -type f -delete
```

-----

#### Upload Command

The command used for the actual upload is:

```
deno run -A jsr:@openneuro/cli upload --dataset ds006081 /Users/liuq13/SPCD_data_BIDS
```