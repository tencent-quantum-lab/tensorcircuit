import OMEinsum
import ArgParse
import JSON

function parse_commandline()
    s = ArgParse.ArgParseSettings()

    @ArgParse.add_arg_table s begin
        "--einsum_json"
            arg_type = String
            default = "einsum.json"
        "--result_json"
            arg_type = String
            default = "opteinsum.json"
        "--sc_target"
            arg_type = Float64
            default = 20.0
        "--beta_start"
            arg_type = Float64
            default = 0.01
        "--beta_step"
            arg_type = Float64
            default = 0.01
        "--beta_stop"
            arg_type = Float64
            default = 15.0
        "--ntrials"
            arg_type = Int
            default = 10
        "--niters"
            arg_type = Int
            default = 50
        "--sc_weight"
            arg_type = Float64
            default = 1.0
        "--rw_weight"
            arg_type = Float64
            default = 0.2
    end

    return ArgParse.parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    # println("Parsed args:")
    # for (arg,val) in parsed_args
    #     println("  $arg  =>  $val")
    # end
    # println(Threads.nthreads())
    contraction_args = JSON.parsefile(parsed_args["einsum_json"])

    inputs = map(Tuple, contraction_args["inputs"])
    output = contraction_args["output"]

    eincode = OMEinsum.EinCode(Tuple(inputs), Tuple(output))

    size_dict = OMEinsum.uniformsize(eincode, 2)
    for (k, v) in contraction_args["size"]
        size_dict[k] = v
    end
    algorithm = OMEinsum.TreeSA(
            sc_target=parsed_args["sc_target"],
            βs=parsed_args["beta_start"]:parsed_args["beta_step"]:parsed_args["beta_stop"],
            ntrials=parsed_args["ntrials"],
            niters=parsed_args["niters"],
            sc_weight=parsed_args["sc_weight"],
            rw_weight=parsed_args["rw_weight"]
        )
    # println(parsed_args["beta_start"]:parsed_args["beta_step"]:parsed_args["beta_stop"])
    # println(algorithm)
    optcode = OMEinsum.optimize_code(eincode, size_dict, algorithm)
    OMEinsum.writejson(parsed_args["result_json"], optcode)
end

main()
