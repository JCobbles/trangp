from multiprocessing import Pool
from reggae.models import transcription

def create_chains(model, args, sample_kwargs, num_chains=4):
    results = list()
    with Pool(num_chains) as p:
        for i in range(num_chains):
            results.append(p.apply_async(run_job, [args, sample_kwargs]))

        res = [result.get() for result in results]


    return res


def run_job(args, sample_kwargs):
    model = transcription.TranscriptionMCMC(*args)
    model.sample(**sample_kwargs)    
    return model
