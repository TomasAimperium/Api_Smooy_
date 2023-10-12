def empresa_schema(empresa) -> dict:
    return {"id": str(empresa["_id"]),
            "nombre_empresa": empresa["nombre_empresa"]}

def empresas_schema(empresas) -> list:
    return [empresa_schema(empresa) for empresa in empresas]